"""
Stage 1 training: binding classification.
Trains, evaluates (train/val), and writes plots/results.

Usage:
    python stage1_train.py --config config_stage1.yaml
"""

import json
from pathlib import Path
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from model import SelectivityModel
from dataset import SelectivityDataset


def compute_class_weights(dataset, indices):
    """Compute inverse frequency weights for class balance."""
    labels = np.array([dataset.examples[idx]['label'] for idx in indices])
    unique_labels, counts = np.unique(labels, return_counts=True)
    weights = np.zeros(len(unique_labels))
    for label, count in zip(unique_labels, counts):
        weights[label] = 1.0 / count if count > 0 else 0
    # Normalize to mean=1
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class StratifiedBatchSampler:
    """Create stratified batches ensuring class/source representation."""
    def __init__(self, dataset, indices, batch_size, shuffle=True, seed=42):
        self.batches = self._create_stratified_batches(dataset, indices, batch_size, shuffle, seed)
    
    def _create_stratified_batches(self, dataset, indices, batch_size, shuffle, seed):
        """Create batches with balanced source/label representation."""
        # Group indices by (source, label)
        groups = {}
        for idx in indices:
            key = (dataset.examples[idx].get('source', 'unknown'), dataset.examples[idx]['label'])
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        
        # Shuffle within each group
        rng = np.random.RandomState(seed)
        for key in groups:
            if shuffle:
                rng.shuffle(groups[key])
        
        # Create batches by cycling through groups
        batches = []
        batch = []
        iterators = {key: iter(group_indices) for key, group_indices in groups.items()}
        keys = list(groups.keys())
        key_idx = 0
        
        while True:
            found_any = False
            for _ in range(len(keys)):
                key = keys[key_idx % len(keys)]
                try:
                    idx = next(iterators[key])
                    batch.append(idx)
                    found_any = True
                except StopIteration:
                    pass
                key_idx += 1
                
                if len(batch) >= batch_size:
                    batches.append(batch)
                    batch = []
            
            if not found_any:
                break
        
        if batch:
            batches.append(batch)
        
        return batches
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)


class StratifiedBatchDataLoader:
    """Wrapper that converts stratified batch indices into tensors."""
    def __init__(self, dataset, batches):
        self.dataset = dataset
        self.batches = batches
    
    def __iter__(self):
        """Yield batches of tensors."""
        for batch_indices in self.batches:
            batch = {
                'ligand_feat': torch.stack([
                    torch.tensor(self.dataset.examples[i]['ligand_feat'], dtype=torch.float32) 
                    for i in batch_indices
                ]),
                'pocket_emb': torch.stack([
                    torch.tensor(self.dataset.examples[i]['pocket_emb'], dtype=torch.float32) 
                    for i in batch_indices
                ]),
                'label': torch.tensor([self.dataset.examples[i]['label'] for i in batch_indices], dtype=torch.long),
                'source': [self.dataset.examples[i].get('source', 'unknown') for i in batch_indices],
                'target': [self.dataset.examples[i].get('target', None) for i in batch_indices],
            }
            yield batch
    
    def __len__(self):
        return len(self.batches)


def hard_batch_triplet_loss_2class(metric_embs, targets, margin=0.3):
    """
    Original 2-class hard-batch triplet loss — used with branch='pre_film'.

    Operates only on PDB examples (passed in as metric_embs).
    Classes: 'PGK1' vs 'PGK2' (selectivity target of the compound).
    Since the projector is pocket-blind (pre-FiLM), the same molecule maps to
    the same point regardless of which pocket it was paired with.

    For each PGK2 anchor: hardest positive = farthest PGK2, hardest negative = closest PGK1.
    For each PGK1 anchor: hardest positive = farthest PGK1, hardest negative = closest PGK2.
    """
    if len(metric_embs) < 2:
        return torch.tensor(0.0, requires_grad=True, device=metric_embs.device)

    dists = torch.cdist(metric_embs, metric_embs, p=2)  # (N, N)

    pgk2_idx = [i for i, t in enumerate(targets) if t == 'PGK2']
    pgk1_idx = [i for i, t in enumerate(targets) if t == 'PGK1']

    if not pgk2_idx or not pgk1_idx:
        return torch.tensor(0.0, requires_grad=True, device=metric_embs.device)

    losses = []
    for a in pgk2_idx:
        pos_indices = [i for i in pgk2_idx if i != a]
        if not pos_indices:
            continue
        d_pos = dists[a][pos_indices].max()
        d_neg = dists[a][pgk1_idx].min()
        losses.append(torch.clamp(d_pos - d_neg + margin, min=0.0))

    for a in pgk1_idx:
        pos_indices = [i for i in pgk1_idx if i != a]
        neg_indices = pgk2_idx
        if not pos_indices:
            d_neg = dists[a][neg_indices].min()
            losses.append(torch.clamp(-d_neg + margin, min=0.0))
            continue
        d_pos = dists[a][pos_indices].max()
        d_neg = dists[a][neg_indices].min()
        losses.append(torch.clamp(d_pos - d_neg + margin, min=0.0))

    if not losses:
        return torch.tensor(0.0, requires_grad=True, device=metric_embs.device)

    return torch.stack(losses).mean()


def hard_batch_triplet_loss(metric_embs, classes, margin=0.3):
    """
    Hard-batch triplet loss on joint (pocket, ligand) embeddings.

    Operates on the full batch (PDB + DECOY alike) using 3 clusters:
      'PGK2_bind' : PGK2 pocket + PGK2-selective binder  (label=1, target=PGK2)
      'PGK1_bind' : PGK1 pocket + PGK1-selective binder  (label=1, target=PGK1)
      'nonbind'   : any pocket  + non-binder             (label=0, DECOY or counter-pocket)

    For each anchor in cluster C:
      positive = farthest other example in C      (hardest positive)
      negative = closest example NOT in C         (hardest negative)
      loss     = max(0, d_pos - d_neg + margin)

    Parameters
    ----------
    metric_embs : (N, 64) L2-normalized joint embeddings
    classes     : list of str, one of 'PGK2_bind' / 'PGK1_bind' / 'nonbind'
    margin      : float

    Returns
    -------
    loss : scalar tensor
    """
    if len(metric_embs) < 2:
        return torch.tensor(0.0, requires_grad=True, device=metric_embs.device)

    dists = torch.cdist(metric_embs, metric_embs, p=2)  # (N, N)

    idx_by_class = {}
    for i, c in enumerate(classes):
        idx_by_class.setdefault(c, []).append(i)

    bind_classes = [c for c in ('PGK2_bind', 'PGK1_bind') if c in idx_by_class]
    # Need at least one bind class present to compute meaningful loss
    if not bind_classes:
        return torch.tensor(0.0, requires_grad=True, device=metric_embs.device)

    losses = []
    all_classes = list(idx_by_class.keys())

    for cls, anchors in idx_by_class.items():
        pos_pool = [i for i in anchors]          # same class
        neg_pool = [i for c, idxs in idx_by_class.items()
                    if c != cls for i in idxs]   # all other classes
        if not neg_pool:
            continue
        for a in anchors:
            pos_indices = [i for i in pos_pool if i != a]
            if not pos_indices:
                # singleton class — only push away from negatives
                d_neg = dists[a][neg_pool].min()
                losses.append(torch.clamp(margin - d_neg, min=0.0))
                continue
            d_pos = dists[a][pos_indices].max()   # hardest positive (farthest same-class)
            d_neg = dists[a][neg_pool].min()      # hardest negative (closest other-class)
            losses.append(torch.clamp(d_pos - d_neg + margin, min=0.0))

    if not losses:
        return torch.tensor(0.0, requires_grad=True, device=metric_embs.device)

    return torch.stack(losses).mean()


def filter_indices_by_source(dataset, sources):
    """Filter dataset indices by source."""
    indices = []
    for idx, example in enumerate(dataset.examples):
        if example.get('source', 'unknown') in sources:
            indices.append(idx)
    return np.array(indices)


def stratified_split_indices(indices, dataset, test_size, seed):
    """Stratified split on indices based on source+label."""
    stratify_keys = [
        f"{dataset.examples[idx].get('source', 'unknown')}_{dataset.examples[idx].get('label', 0)}"
        for idx in indices
    ]
    
    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_keys,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
        )
    
    return train_idx, val_idx


def load_checkpoint_into_model(checkpoint_path, model):
    """Load checkpoint weights into model."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded checkpoint from {checkpoint_path}")
    return model


def train_epoch(model, dataloader, optimizer, criterion, device, class_weights=None, metric_weight=0.0, metric_margin=0.3):
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

        # Compute loss with class weights if provided
        if class_weights is not None:
            class_weights_device = class_weights.to(device)
            sample_weights = class_weights_device[labels.long()]
            bce_loss = nn.BCEWithLogitsLoss(reduction='none')
            loss = bce_loss(logits, labels.unsqueeze(1).float())
            loss = (loss * sample_weights.unsqueeze(1)).mean()
        else:
            loss = criterion(logits, labels.unsqueeze(1).float())

        # Metric triplet loss — behaviour depends on branch setting
        if metric_weight > 0.0 and getattr(model, 'use_metric_head', False):
            sources_batch = batch.get('source', [])
            targets_batch = batch.get('target', [])
            labels_batch  = batch['label'].cpu().tolist()

            if getattr(model, 'metric_branch', 'post_film') == 'pre_film':
                # PRE-FILM: original 2-class PDB-only loss.
                # Uses ALL PDB examples (both pockets, both labels) — pocket-blind so
                # counter-pocket and binding-pocket of same compound produce identical embedding.
                # Classes: 'PGK1' vs 'PGK2' target label of the compound.
                pdb_mask = [i for i, src in enumerate(sources_batch) if src == 'PDB']
                if len(pdb_mask) >= 2:
                    pdb_indices = torch.tensor(pdb_mask, dtype=torch.long, device=device)
                    pdb_targets = [targets_batch[i] for i in pdb_mask]  # 'PGK1' or 'PGK2'
                    _, _, metric_embs = model(
                        ligand_feat[pdb_indices], pocket_emb[pdb_indices], return_metric_emb=True
                    )
                    triplet_loss = hard_batch_triplet_loss_2class(metric_embs, pdb_targets, margin=metric_margin)
                    loss = loss + metric_weight * triplet_loss
            else:
                # POST-FILM: 3-class full-batch loss.
                # Classes: 'PGK2_bind' / 'PGK1_bind' / 'nonbind' (DECOY + counter-pockets).
                # Pocket-aware — requires pdb_oversample_factor > 1 to work well.
                metric_classes = []
                for src, tgt, lbl in zip(sources_batch, targets_batch, labels_batch):
                    if src == 'PDB' and int(lbl) == 1 and tgt == 'PGK2':
                        metric_classes.append('PGK2_bind')
                    elif src == 'PDB' and int(lbl) == 1 and tgt == 'PGK1':
                        metric_classes.append('PGK1_bind')
                    else:
                        metric_classes.append('nonbind')
                _, _, metric_embs = model(ligand_feat, pocket_emb, return_metric_emb=True)
                triplet_loss = hard_batch_triplet_loss(metric_embs, metric_classes, margin=metric_margin)
                loss = loss + metric_weight * triplet_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(probs.cpu().detach().numpy().flatten())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, len(dataloader))
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else np.nan
    acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)

    return avg_loss, auc, acc


def validate(model, dataloader, criterion, device, class_weights=None, return_sources=False):
    """Validate on a dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_sources = []

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
            all_sources.extend(batch.get('source', ['unknown'] * len(labels)))

    avg_loss = total_loss / max(1, len(dataloader))
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else np.nan
    acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)

    if return_sources:
        return avg_loss, auc, acc, all_preds, all_labels, all_sources
    return avg_loss, auc, acc, all_preds, all_labels


def stratified_split(dataset, test_size, seed):
    """Split dataset into train/val with stratification on source+label."""
    indices = np.arange(len(dataset))
    stratify_keys = [
        f"{ex.get('source', 'unknown')}_{ex.get('label', 0)}"
        for ex in dataset.examples
    ]

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_keys,
        )
    except ValueError:
        labels = [ex.get('label', 0) for ex in dataset.examples]
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )

    return Subset(dataset, train_idx), Subset(dataset, val_idx), train_idx


def train_stage1(model, train_dataset, val_dataset, config, device, class_weights=None, use_stratified_sampler=False, use_weighted_sampler=False, pdb_oversample_factor=1.0, validation_metric_config=None, metric_weight=0.0, metric_margin=0.3):
    """Stage 1: Binding classification."""
    print("\n" + "=" * 60)
    print("STAGE 1: Binding Classification")
    print("=" * 60)

    # Use weighted random sampler if requested (oversamples PDB for better representation)
    if use_weighted_sampler:
        """Create weighted sampler that oversamples PDB samples."""
        if isinstance(train_dataset, Subset):
            indices = np.array(train_dataset.indices)
            base_dataset = train_dataset.dataset
        else:
            indices = np.arange(len(train_dataset))
            base_dataset = train_dataset
        
        # Compute sample weights: PDB gets higher probability
        weights = np.ones(len(indices))
        for i, idx in enumerate(indices):
            source = base_dataset.examples[idx].get('source', 'unknown')
            if source == 'PDB':
                weights[i] = pdb_oversample_factor
        
        weights = weights / weights.sum()  # Normalize
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=sampler,
            num_workers=0,
        )
        print(f"  Using WeightedRandomSampler with PDB oversample factor: {pdb_oversample_factor:.1f}x")
    
    # Use stratified batch sampler if requested (for handling class imbalance)
    elif use_stratified_sampler:
        # Get indices from Subset if needed
        if isinstance(train_dataset, Subset):
            train_indices = np.array(train_dataset.indices)
            base_dataset = train_dataset.dataset
        else:
            train_indices = np.arange(len(train_dataset))
            base_dataset = train_dataset
        
        sampler = StratifiedBatchSampler(
            base_dataset,
            train_indices,
            batch_size=config['batch_size'],
            shuffle=True,
            seed=config['seed']
        )
        
        train_loader = StratifiedBatchDataLoader(base_dataset, sampler.batches)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config['lr_stage1'],
        weight_decay=config['weight_decay'],
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_epoch = 0

    history = {
        'epoch': [],
        'train_loss': [],
        'train_auc': [],
        'train_acc': [],
        'val_loss': [],
        'val_auc': [],
        'val_acc': [],
    }

    for epoch in range(config['epochs_stage1']):
        train_loss, train_auc, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, class_weights=class_weights,
            metric_weight=metric_weight, metric_margin=metric_margin
        )

        val_loss, val_auc, val_acc, val_preds, val_labels, val_sources = validate(
            model, val_loader, criterion, device, return_sources=True
        )
        
        # Get predictions on training set to compute PDB metrics across ALL PDB samples (train + val)
        train_loss_eval, _, _, train_preds, train_labels, train_sources = validate(
            model, train_loader, criterion, device, return_sources=True
        )
        
        # Combine train + val predictions for metrics computation
        all_preds = np.concatenate([np.array(train_preds), np.array(val_preds)])
        all_labels = np.concatenate([np.array(train_labels), np.array(val_labels)])
        all_sources = np.concatenate([np.array(train_sources), np.array(val_sources)])
        
        # Compute PDB-specific metrics across ALL PDB samples (both train and val)
        pdb_mask = all_sources == 'PDB'
        
        if pdb_mask.sum() > 0:
            pdb_labels = all_labels[pdb_mask]
            pdb_preds = all_preds[pdb_mask]
            # For PDB AUC, only compute if we have both classes
            if len(np.unique(pdb_labels)) > 1:
                pdb_auc = roc_auc_score(pdb_labels, pdb_preds)
            else:
                pdb_auc = np.nan
            pdb_acc = accuracy_score(pdb_labels, pdb_preds > 0.5)
            pdb_sensitivity = (pdb_preds[pdb_labels == 1] > 0.5).sum() / max(1, (pdb_labels == 1).sum())
        else:
            pdb_auc = np.nan
            pdb_acc = np.nan
            pdb_sensitivity = np.nan
        
        # Compute validation metric for best model selection (PDB + subset of DECOY)
        if validation_metric_config is not None and validation_metric_config.get('include_all_pdb', True):
            decoy_fraction = validation_metric_config.get('decoy_fraction', 0.0)
            
            # Start with all PDB samples
            subset_mask = all_sources == 'PDB'
            
            # Add a fraction of DECOY samples
            if decoy_fraction > 0:
                decoy_mask = all_sources == 'DECOY'
                decoy_indices = np.where(decoy_mask)[0]
                
                # Sample a fraction of DECOY indices
                n_decoy_to_include = int(len(decoy_indices) * decoy_fraction)
                if n_decoy_to_include > 0:
                    np.random.seed(config['seed'] + epoch)  # Deterministic but varies by epoch
                    selected_decoy = np.random.choice(decoy_indices, size=n_decoy_to_include, replace=False)
                    subset_mask[selected_decoy] = True
            
            subset_labels = all_labels[subset_mask]
            subset_preds = all_preds[subset_mask]
            
            if len(np.unique(subset_labels)) > 1 and len(subset_labels) > 0:
                stopping_metric = roc_auc_score(subset_labels, subset_preds)
                stopping_metric_name = f"PDB+DECOY({int(decoy_fraction*100)}%) AUC"
            else:
                stopping_metric = pdb_auc
                stopping_metric_name = "PDB AUC"
        else:
            # Default: use pure PDB AUC
            stopping_metric = pdb_auc
            stopping_metric_name = "PDB AUC"

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}/{config['epochs_stage1']}")
        print(f"  Train: loss={train_loss:.4f}, AUC={train_auc:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, AUC={val_auc:.4f}, acc={val_acc:.4f}")
        if not np.isnan(pdb_auc):
            print(f"  PDB (all {pdb_mask.sum()} samples):   AUC={pdb_auc:.4f}, acc={pdb_acc:.4f}, sensitivity={pdb_sensitivity:.4f}")

        # Use stopping metric for best model selection
        if not np.isnan(stopping_metric) and stopping_metric > best_val_auc:
            best_val_auc = stopping_metric
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'pdb_auc': pdb_auc,
                'stopping_metric': stopping_metric,
            }, config['checkpoint_stage1'])
            print(f"  -> Saved best model ({stopping_metric_name}={stopping_metric:.4f})")

    print(f"\nBest Stage 1 model: epoch {best_epoch}, val AUC={best_val_auc:.4f}")

    history_path = config['checkpoint_stage1'].replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    return best_val_auc, history, best_epoch


def evaluate_split(model, dataset, stage_name, split_name, checkpoint_path, results_dir, device):
    """Evaluate a model on a dataset split and save results."""
    print(f"\nEvaluating {stage_name} ({split_name})...")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    all_preds = []
    all_probs = []
    all_labels = []
    all_sources = []

    with torch.no_grad():
        for batch in data_loader:
            ligand_feat = batch['ligand_feat'].to(device)
            pocket_emb = batch['pocket_emb'].to(device)
            labels = batch['label'].to(device)

            logits, probs = model(ligand_feat, pocket_emb)

            all_preds.extend((probs > 0.5).cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
            all_sources.extend(batch.get('source', ['unknown'] * len(labels)))

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else np.nan
    acc = accuracy_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    results_df = {
        'true_label': all_labels,
        'predicted_label': all_preds,
        'predicted_probability': all_probs,
        'source': all_sources,
    }

    stage_key = stage_name.lower().replace(" ", "_")
    split_key = split_name.lower()
    preds_path = results_dir / f"{stage_key}_{split_key}_predictions.csv"
    metrics_path = results_dir / f"{stage_key}_{split_key}_metrics.json"

    import pandas as pd
    pd.DataFrame(results_df).to_csv(preds_path, index=False)
    print(f"  Saved predictions to {preds_path}")

    # Per-source metrics
    per_source = {}
    unique_sources = sorted(set(all_sources))
    for source in unique_sources:
        mask = np.array(all_sources) == source
        src_labels = all_labels[mask]
        src_preds = all_preds[mask]
        src_probs = all_probs[mask]

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
        'checkpoint_path': str(checkpoint_path),
        'split': split_name,
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

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    print(f"\n{stage_name} {split_name} Results:")
    print(f"  Samples: {len(all_labels)} (pos: {int(np.sum(all_labels))}, neg: {int(len(all_labels) - np.sum(all_labels))})")
    print(f"  AUC-ROC: {auc if not np.isnan(auc) else 'N/A'}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return metrics


def evaluate_pdb_predictions(model, dataset, checkpoint_path, results_dir, device):
    """
    Run the best checkpoint on all PDB ligands with BOTH pockets and write
    TWO rows per ligand (one per pocket) — 30 rows total for 15 compounds.

    Output columns
    --------------
    mol_id, ligand_id, description, smiles,
    pocket,                  # 'PGK1' or 'PGK2'
    true_label,              # 1 = binds this pocket, 0 = does not
    true_target,             # which pocket the compound actually selects
    p_bind,                  # model P(bind | this pocket)
    pred_label,              # 1 if p_bind > 0.5
    correct                  # pred_label == true_label

    Summary CSV (one row per compound)
    -----------------------------------
    p_bind_pgk1, p_bind_pgk2, pred_selectivity_score, pred_target, correct_selectivity
    """
    import pandas as pd

    print("\n" + "=" * 60)
    print("PDB LIGAND PREDICTIONS  (15 ligands × 2 pockets = 30 rows)")
    print("=" * 60)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    pgk1_mean = torch.tensor(dataset.pgk1_mean, dtype=torch.float32).unsqueeze(0).to(device)
    pgk2_mean = torch.tensor(dataset.pgk2_mean, dtype=torch.float32).unsqueeze(0).to(device)

    pdb_meta = dataset.metadata[dataset.metadata['source'] == 'PDB'].copy()

    long_rows    = []   # 2 rows per ligand (one per pocket)
    summary_rows = []   # 1 row per ligand

    for _, row in pdb_meta.iterrows():
        smi = row['smiles']
        if pd.isna(smi) or smi not in dataset.ligand_features:
            continue

        feat = torch.tensor(dataset.ligand_features[smi], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            _, p1 = model(feat, pgk1_mean)
            _, p2 = model(feat, pgk2_mean)

        p_pgk1 = float(p1.item())
        p_pgk2 = float(p2.item())
        true_target = row.get('target', '')

        base = {
            'mol_id'     : row.get('mol_id', ''),
            'ligand_id'  : row.get('ligand_id', ''),
            'description': row.get('description', ''),
            'smiles'     : smi,
            'true_target': true_target,
        }

        # Row 1: evaluated against PGK1 pocket
        tl_pgk1 = 1 if true_target == 'PGK1' else 0
        long_rows.append({**base,
            'pocket'     : 'PGK1',
            'true_label' : tl_pgk1,
            'p_bind'     : round(p_pgk1, 4),
            'pred_label' : int(p_pgk1 > 0.5),
            'correct'    : int(p_pgk1 > 0.5) == tl_pgk1,
        })

        # Row 2: evaluated against PGK2 pocket
        tl_pgk2 = 1 if true_target == 'PGK2' else 0
        long_rows.append({**base,
            'pocket'     : 'PGK2',
            'true_label' : tl_pgk2,
            'p_bind'     : round(p_pgk2, 4),
            'pred_label' : int(p_pgk2 > 0.5),
            'correct'    : int(p_pgk2 > 0.5) == tl_pgk2,
        })

        # Summary row
        score = p_pgk2 - p_pgk1
        pred_target = 'PGK2' if score > 0 else 'PGK1'
        summary_rows.append({**base,
            'true_selectivity'      : row.get('selectivity', ''),
            'p_bind_pgk1'           : round(p_pgk1, 4),
            'p_bind_pgk2'           : round(p_pgk2, 4),
            'pred_selectivity_score': round(score, 4),
            'pred_target'           : pred_target,
            'correct_selectivity'   : pred_target == true_target,
        })

    df_long    = pd.DataFrame(long_rows).sort_values(['true_target', 'mol_id', 'pocket']).reset_index(drop=True)
    df_summary = pd.DataFrame(summary_rows).sort_values(['true_target', 'mol_id']).reset_index(drop=True)

    out_long    = Path(results_dir) / 'pdb_ligand_predictions.csv'
    out_summary = Path(results_dir) / 'pdb_ligand_predictions_summary.csv'
    df_long.to_csv(out_long, index=False)
    df_summary.to_csv(out_summary, index=False)

    # ── Console: 30-row table ─────────────────────────────────────────────────
    print(df_long[['ligand_id', 'description', 'true_target', 'pocket',
                   'true_label', 'p_bind', 'pred_label', 'correct']].to_string(index=False))

    # ── Accuracy breakdown ────────────────────────────────────────────────────
    print("\n  ── Binding prediction accuracy (per pocket) ──")
    for pocket in ['PGK1', 'PGK2']:
        sub = df_long[df_long['pocket'] == pocket]
        acc = sub['correct'].mean()
        print(f"    {pocket} pocket: {sub['correct'].sum()}/{len(sub)} correct  ({100*acc:.1f}%)")
    print(f"    Overall (30 rows): {df_long['correct'].sum()}/{len(df_long)} correct  "
          f"({100*df_long['correct'].mean():.1f}%)")

    print("\n  ── Selectivity classification accuracy (15 ligands) ──")
    for tgt in ['PGK1', 'PGK2']:
        sub = df_summary[df_summary['true_target'] == tgt]
        acc = sub['correct_selectivity'].mean() if len(sub) else float('nan')
        print(f"    {tgt}: {sub['correct_selectivity'].sum()}/{len(sub)} correct  ({100*acc:.1f}%)")
    overall = df_summary['correct_selectivity'].mean()
    print(f"    Overall: {df_summary['correct_selectivity'].sum()}/{len(df_summary)} correct  ({100*overall:.1f}%)")

    print(f"\n  Saved → {out_long}")
    print(f"  Saved → {out_summary}")

    return df_long, df_summary


def plot_training_history(history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['epoch'], history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['epoch'], history['val_loss'], label='Validation', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Stage 1: Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['epoch'], history['train_auc'], label='Train', marker='o')
    axes[1].plot(history['epoch'], history['val_auc'], label='Validation', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC-ROC')
    axes[1].set_title('Stage 1: AUC-ROC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.8, 1.05])

    axes[2].plot(history['epoch'], history['train_acc'], label='Train', marker='o')
    axes[2].plot(history['epoch'], history['val_acc'], label='Validation', marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Stage 1: Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0.8, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved training plot to {save_path}")
    plt.close()


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


def main():
    parser = argparse.ArgumentParser(description='Stage 1 training: binding classification')
    parser.add_argument('--config', type=str, default='config_stage1.yaml',
                       help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from: {args.config}\n")
    
    # Extract paths and parameters
    version = config.get('version', 'v0')
    checkpoint_dir = Path(config['output']['checkpoint_dir']) / version
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Device selection
    device_config = config['training']['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    
    print(f"Using device: {device}\n")
    
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    metric_cfg = config.get('metric_learning', {})
    use_metric_head = metric_cfg.get('enabled', False)
    metric_proj_dim = metric_cfg.get('proj_dim', 64)
    metric_branch   = metric_cfg.get('branch', 'post_film')
    model = SelectivityModel(
        use_metric_head=use_metric_head,
        metric_proj_dim=metric_proj_dim,
        metric_branch=metric_branch,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if use_metric_head:
        print(f"Metric head enabled (proj_dim={metric_proj_dim}, branch={metric_branch})\n")
    
    # Check if three-stage training is enabled
    three_stage_config = config.get('three_stage', {})
    three_stage_enabled = three_stage_config.get('enabled', False)
    
    if three_stage_enabled:
        print("="*60)
        print("TWO-STAGE TRAINING PIPELINE")
        print("="*60)
        
        # ===== STAGE 1A: Train on DEL + DECOY =====
        print("\n[STAGE 1A] Training on DEL + DECOY sources...")
        
        checkpoint_path_1a = checkpoint_dir / config['output'].get('checkpoint_name_stage1a', 'stage1a_best.pt')
        history_path_1a = checkpoint_dir / config['output'].get('history_name_stage1a', 'stage1a_best_history.json')
        
        print(f"\nLoading Stage 1 data ({config['training']['stage']})...")
        full_dataset = SelectivityDataset(stage=config['training']['stage'], csv_path=config['input']['ligand_csv'])
        
        # Filter to only DEL + DECOY
        stage1a_sources = three_stage_config.get('stage1a_sources', ['DEL', 'DECOY'])
        stage1a_indices = filter_indices_by_source(full_dataset, stage1a_sources)
        print(f"Filtered to {len(stage1a_indices)} examples from {stage1a_sources}")
        
        # Stratified split for Stage 1A
        train_idx_1a, val_idx_1a = stratified_split_indices(
            stage1a_indices,
            full_dataset,
            test_size=config['training']['val_split'],
            seed=config['training']['seed'],
        )
        
        train_dataset_1a = Subset(full_dataset, train_idx_1a)
        val_dataset_1a = Subset(full_dataset, val_idx_1a)
        
        print(f"Stage 1A - Train: {len(train_dataset_1a)}, Val: {len(val_dataset_1a)}")
        
        # Train Stage 1A
        train_config_1a = {
            'batch_size': config['training']['batch_size'],
            'lr_stage1': three_stage_config.get('stage1a_learning_rate', config['training']['learning_rate']),
            'epochs_stage1': three_stage_config.get('stage1a_epochs', 20),
            'weight_decay': config['training']['weight_decay'],
            'seed': config['training']['seed'],
            'checkpoint_stage1': str(checkpoint_path_1a),
        }
        
        auc_1a, history_1a, best_epoch_1a = train_stage1(model, train_dataset_1a, val_dataset_1a, train_config_1a, device)
        
        # Evaluate Stage 1A separately
        print(f"\n[STAGE 1A EVALUATION]")
        if config['evaluation']['save_train_results']:
            evaluate_split(model, train_dataset_1a, 'Stage 1A', 'Train', str(checkpoint_path_1a), results_dir, device)
        
        if config['evaluation']['save_val_results']:
            evaluate_split(model, val_dataset_1a, 'Stage 1A', 'Val', str(checkpoint_path_1a), results_dir, device)
        
        if config['evaluation']['per_source_confusion']:
            if config['evaluation']['save_train_results']:
                confusion_plot_1a_train = results_dir / 'stage_1a_train_confusion_matrices.png'
                plot_per_source_confusion_matrices(evaluate_split(model, train_dataset_1a, 'Stage 1A', 'Train', str(checkpoint_path_1a), results_dir, device), 'Stage 1A', 'Train', confusion_plot_1a_train)
            if config['evaluation']['save_val_results']:
                confusion_plot_1a_val = results_dir / 'stage_1a_val_confusion_matrices.png'
                plot_per_source_confusion_matrices(evaluate_split(model, val_dataset_1a, 'Stage 1A', 'Val', str(checkpoint_path_1a), results_dir, device), 'Stage 1A', 'Val', confusion_plot_1a_val)
        
        # ===== STAGE 1B: Fine-tune on DECOY + PDB, validate on stratified DECOY+PDB =====
        print("\n[STAGE 1B] Fine-tuning on DECOY + PDB, validating on stratified DECOY+PDB...")
        
        checkpoint_path_1b = checkpoint_dir / config['output'].get('checkpoint_name_stage1b', 'stage1b_best.pt')
        history_path_1b = checkpoint_dir / config['output'].get('history_name_stage1b', 'stage1b_best_history.json')
        
        # Load Stage 1A checkpoint into a fresh model
        model1b = SelectivityModel().to(device)
        load_checkpoint_into_model(str(checkpoint_path_1a), model1b)
        
        # Filter to DECOY + PDB for training and validation (will be stratified split)
        stage1b_sources = three_stage_config.get('stage1b_train_sources', ['DECOY', 'PDB'])
        stage1b_all_indices = filter_indices_by_source(full_dataset, stage1b_sources)
        
        # Stratified split of DECOY+PDB into train/val
        train_idx_1b, val_idx_1b = stratified_split_indices(
            stage1b_all_indices,
            full_dataset,
            test_size=config['training']['val_split'],
            seed=config['training']['seed'],
        )
        
        print(f"Filtered to {len(stage1b_all_indices)} examples from {stage1b_sources}")
        print(f"  Train: {len(train_idx_1b)}, Val: {len(val_idx_1b)} (80/20 stratified split)")
        
        train_dataset_1b = Subset(full_dataset, train_idx_1b)
        val_dataset_1b = Subset(full_dataset, val_idx_1b)
        
        # Separate external test set on DEL
        test_idx_1b = filter_indices_by_source(full_dataset, three_stage_config.get('stage1b_test_sources', ['DEL']))
        test_dataset_1b = Subset(full_dataset, test_idx_1b)
        print(f"  External test (DEL): {len(test_dataset_1b)}")
        
        # Compute class weights for Stage 1B training (DECOY + PDB imbalance)
        print(f"\nComputing class weights for Stage 1B (DECOY+PDB)...")
        class_weights_1b = compute_class_weights(full_dataset, train_idx_1b)
        print(f"  Class weights: {class_weights_1b.numpy()}")
        
        # Count PDB/DECOY in training
        pdb_count = sum(1 for idx in train_idx_1b if full_dataset.examples[idx].get('source') == 'PDB')
        decoy_count = sum(1 for idx in train_idx_1b if full_dataset.examples[idx].get('source') == 'DECOY')
        print(f"  Stage 1B Train composition: PDB={pdb_count}, DECOY={decoy_count}")
        
        print(f"Stage 1B - Train: {len(train_dataset_1b)}, Val: {len(val_dataset_1b)}")
        
        # Train Stage 1B with class weights and stratified sampling
        train_config_1b = {
            'batch_size': config['training']['batch_size'],
            'lr_stage1': three_stage_config.get('stage1b_learning_rate', config['training']['learning_rate']),
            'epochs_stage1': three_stage_config.get('stage1b_epochs', 30),
            'weight_decay': config['training']['weight_decay'],
            'seed': config['training']['seed'],
            'checkpoint_stage1': str(checkpoint_path_1b),
        }
        
        auc_1b, history_1b, best_epoch_1b = train_stage1(
            model1b, train_dataset_1b, val_dataset_1b, train_config_1b, device,
            class_weights=class_weights_1b,
            use_stratified_sampler=False,
            use_weighted_sampler=True,
            pdb_oversample_factor=three_stage_config.get('stage1b_pdb_oversample_factor', 1.0),
            validation_metric_config=three_stage_config.get('stage1b_validation_metric', None)
        )
        
        # Use Stage 1B as final model for evaluation
        final_checkpoint = str(checkpoint_path_1b)
        model_for_eval = model1b
        
        print(f"\n[SUMMARY] Stage 1A val AUC: {auc_1a:.4f}, Stage 1B val AUC: {auc_1b:.4f}")
        
    else:
        # Standard single-stage training
        checkpoint_path = checkpoint_dir / config['output']['checkpoint_name']
        history_path = checkpoint_dir / config['output']['history_name']
        
        train_config = {
            'batch_size': config['training']['batch_size'],
            'lr_stage1': config['training']['learning_rate'],
            'epochs_stage1': config['training']['epochs'],
            'weight_decay': config['training']['weight_decay'],
            'seed': config['training']['seed'],
            'checkpoint_stage1': str(checkpoint_path),
        }
        
        print(f"\nLoading Stage 1 data ({config['training']['stage']})...")
        full_dataset = SelectivityDataset(stage=config['training']['stage'], csv_path=config['input']['ligand_csv'])
        
        # Keep ALL PDB examples in train — only split DECOY for val (monitoring overfitting)
        # Stage 2 LOO is the real PDB evaluation
        pdb_idx = filter_indices_by_source(full_dataset, ['PDB'])
        decoy_idx = filter_indices_by_source(full_dataset, ['DECOY'])
        
        decoy_train_idx, decoy_val_idx = train_test_split(
            decoy_idx,
            test_size=config['training']['val_split'],
            random_state=train_config['seed'],
        )
        
        train_idx = np.concatenate([pdb_idx, decoy_train_idx])
        val_idx = decoy_val_idx
        
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        
        pdb_count = len(pdb_idx)
        print(f"Stage 1 - Train: {len(train_dataset)} (PDB={pdb_count}, DECOY={len(decoy_train_idx)}), Val: {len(val_dataset)} (DECOY-only)")
        
        # Compute class weights to handle imbalance
        class_weights = compute_class_weights(full_dataset, train_idx)
        print(f"Class weights: neg={class_weights[0]:.4f}, pos={class_weights[1]:.4f}")
        
        metric_weight = metric_cfg.get('weight', 0.0) if metric_cfg.get('enabled', False) else 0.0
        pdb_oversample_factor = config['training'].get('pdb_oversample_factor', 1.0)
        use_weighted = pdb_oversample_factor > 1.0
        if use_weighted:
            print(f"PDB oversample factor: {pdb_oversample_factor:.1f}x (WeightedRandomSampler)")
        metric_margin = metric_cfg.get('margin', 0.3)
        print(f"Metric margin: {metric_margin}")
        auc_1, history, best_epoch_1 = train_stage1(model, train_dataset, val_dataset, train_config, device,
                                                     class_weights=class_weights,
                                                     use_weighted_sampler=use_weighted,
                                                     pdb_oversample_factor=pdb_oversample_factor,
                                                     metric_weight=metric_weight,
                                                     metric_margin=metric_margin)
        
        final_checkpoint = str(checkpoint_path)
        model_for_eval = model

    # Evaluation and plots (using final model)
    if config['evaluation']['save_train_results']:
        # For three-stage, evaluate Stage 1B on training set (DECOY+PDB)
        if three_stage_enabled:
            train_metrics = evaluate_split(model_for_eval, train_dataset_1b, 'Stage 1B', 'Train (DECOY+PDB)', final_checkpoint, results_dir, device)
        else:
            train_metrics = evaluate_split(model_for_eval, train_dataset, 'Stage 1', 'Train', final_checkpoint, results_dir, device)
        
        if config['evaluation']['per_source_confusion']:
            confusion_plot = results_dir / ('stage_1b_train_confusion_matrices.png' if three_stage_enabled else 'stage_1_train_confusion_matrices.png')
            plot_per_source_confusion_matrices(train_metrics, 'Stage 1' + ('B' if three_stage_enabled else ''), 'Train' if three_stage_enabled else 'Train', confusion_plot)
    
    if config['evaluation']['save_val_results']:
        # For three-stage, evaluate Stage 1B on validation set (DECOY+PDB)
        if three_stage_enabled:
            val_metrics = evaluate_split(model_for_eval, val_dataset_1b, 'Stage 1B', 'Val (DECOY+PDB)', final_checkpoint, results_dir, device)
        else:
            val_metrics = evaluate_split(model_for_eval, val_dataset, 'Stage 1', 'Val', final_checkpoint, results_dir, device)
        
        if config['evaluation']['per_source_confusion']:
            confusion_plot = results_dir / ('stage_1b_val_confusion_matrices.png' if three_stage_enabled else 'stage_1_val_confusion_matrices.png')
            plot_per_source_confusion_matrices(val_metrics, 'Stage 1' + ('B' if three_stage_enabled else ''), 'Val' if three_stage_enabled else 'Val', confusion_plot)
    
    # External test set evaluation (for three-stage, test on external DEL after training)
    if three_stage_enabled:
        print(f"\n[EXTERNAL TEST] Evaluating Stage 1B on external DEL test set...")
        test_metrics = evaluate_split(model_for_eval, test_dataset_1b, 'Stage 1B', 'Test (external DEL)', final_checkpoint, results_dir, device)
        if config['evaluation']['per_source_confusion']:
            confusion_plot = results_dir / 'stage_1b_test_confusion_matrices.png'
            plot_per_source_confusion_matrices(test_metrics, 'Stage 1B', 'Test (external DEL)', confusion_plot)
    
    if config['evaluation']['plot_training_curves']:
        if three_stage_enabled:
            plot_training_history(history_1b, results_dir / 'stage_1b_training_curves.png')
        else:
            plot_training_history(history, results_dir / 'stage_1_training_curves.png')

    # ── PDB predictions vs ground truth ──────────────────────────────────────
    evaluate_pdb_predictions(model_for_eval, full_dataset, final_checkpoint, results_dir, device)

    print("\nTraining complete.")
    print(f"Best model checkpoint: {final_checkpoint}")
    if three_stage_enabled:
        print(f"Best Stage 1B model found at epoch: {best_epoch_1b}")
    else:
        print(f"Best model found at epoch: {best_epoch_1}")
    print(f"Results saved to {results_dir}")


if __name__ == '__main__':
    main()
