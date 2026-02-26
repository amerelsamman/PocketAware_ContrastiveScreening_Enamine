"""
Screen Enamine kinase library for PGK1/PGK2 selectivity.

Pipeline
--------
1. Load Enamine CSV (SMILES)
2. Check if embeddings cached; if not, compute them
3. Load Stage 1B checkpoint
4. For each compound:
   - Forward pass with PGK1_mean → p_bind_pgk1
   - Forward pass with PGK2_mean → p_bind_pgk2
   - Compute selectivity: Δ = p_bind_pgk2 - p_bind_pgk1
5. Rank by selectivity and save results

Usage
-----
    conda activate unimol
    python screen_enamine.py --config config_screen_enamine.yaml
"""

import sys
print("[screen_enamine] Python started, loading imports...", flush=True)

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import time
import argparse
import yaml
import contextlib
import logging
print("[screen_enamine] Loading torch...", flush=True)
import torch
print("[screen_enamine] Loading RDKit...", flush=True)
from rdkit import Chem
from rdkit.Chem import AllChem

# Suppress verbose logging from unimol_tools and other libraries
logging.getLogger('unimol_tools').setLevel(logging.ERROR)
logging.getLogger('unimol_tools.tasks').setLevel(logging.ERROR)
logging.getLogger('unimol_tools.tasks.trainer').setLevel(logging.ERROR)

print("[screen_enamine] Loading model...", flush=True)
from model import SelectivityModel
print("[screen_enamine] All imports done. Starting...", flush=True)

# NOTE: UniMolRepr is imported lazily below
#       because it takes 2-3 min to load and would freeze terminal silently


# ─────────────────────────────────────────────────────────────────────────────
# Conformer generation (same as prepare_ligand_features.py)
# ─────────────────────────────────────────────────────────────────────────────

def smiles_to_conformer(smiles):
    """
    Generate 3D conformer for a SMILES string WITHOUT explicit hydrogens.
    Try ETKDG first; if that fails, use random 3D coords.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=False)
        
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        
        if mol.GetNumConformers() == 0:
            return None
        
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass
        
        return mol
    except Exception as e:
        return None


def mol_to_unimol_input(mol):
    """
    Convert RdKit Mol with conformer to Uni-Mol input (atoms + coords).
    """
    if mol.GetNumConformers() == 0:
        return None, None
    
    conf = mol.GetConformer()
    atoms = []
    coords = []
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:  # Skip H
            continue
        atoms.append(atom.GetSymbol())
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    
    if not atoms:
        return None, None
    
    return atoms, np.array(coords, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding computation
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def suppress_output():
    """Suppress stdout, stderr, and logging temporarily."""
    # Suppress logging
    log_level = logging.root.level
    logging.root.setLevel(logging.CRITICAL)
    for logger_name in ['unimol_tools', 'unimol_tools.tasks', 'unimol_tools.tasks.trainer']:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    
    # Suppress stdout/stderr
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # Restore logging
            logging.root.setLevel(log_level)
            for logger_name in ['unimol_tools', 'unimol_tools.tasks', 'unimol_tools.tasks.trainer']:
                logging.getLogger(logger_name).setLevel(logging.ERROR)

def compute_embeddings_batch(smiles_list, mol_repr, batch_size=32, verbose=True):
    """
    Compute Uni-Mol embeddings for a list of SMILES.
    
    Returns
    -------
    embeddings_dict : dict {smiles: embedding (512,)}
    failed_smiles : list of (smiles, reason)
    """
    embeddings_dict = {}
    failed_smiles = []
    
    for idx, smiles in enumerate(smiles_list):
        try:
            mol = smiles_to_conformer(smiles)
            if mol is None:
                failed_smiles.append((smiles, "conformer_failed"))
            else:
                atoms, coords = mol_to_unimol_input(mol)
                if atoms is None or coords is None:
                    failed_smiles.append((smiles, "atomization_failed"))
                else:
                    # Suppress UniMol's verbose output
                    with suppress_output():
                        result = mol_repr.get_repr({'atoms': [atoms], 'coordinates': [coords]})
                    embedding = np.array(result)[0]
                    embeddings_dict[smiles] = embedding
        except Exception as e:
            failed_smiles.append((smiles, f"error: {str(e)[:50]}"))
        
        # Print progress every 1000 molecules
        if verbose and (idx + 1) % 1000 == 0:
            print(f"  Embedded {idx + 1:,} / {len(smiles_list):,}  (failed so far: {len(failed_smiles)})", flush=True)
    
    return embeddings_dict, failed_smiles


# ─────────────────────────────────────────────────────────────────────────────
# Screening
# ─────────────────────────────────────────────────────────────────────────────

def screen_compounds(config_path='config_screen_enamine.yaml'):
    """
    Main screening pipeline.
    """
    print("\n" + "="*70, flush=True)
    print("INITIALIZING: Screen Enamine Kinase Library for PGK1/PGK2 Selectivity", flush=True)
    print("="*70 + "\n", flush=True)
    
    # Load config
    print(f"Loading config from {config_path}...", flush=True)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded\n", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n", flush=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Load Enamine CSV
    # ─────────────────────────────────────────────────────────────────────────
    
    print("STAGE 1: Loading Enamine Compound Library", flush=True)
    
    enamine_csv = Path(config['input']['enamine_csv'])
    print(f"  Loading Enamine compounds from {enamine_csv}...", flush=True)
    df_enamine = pd.read_csv(enamine_csv)
    
    smiles_col = config['input'].get('smiles_column', 'smiles')
    df_enamine = df_enamine.reset_index(drop=True)  # ensure clean 0-based index
    enamine_smiles_series = df_enamine[smiles_col].dropna()
    enamine_smiles = enamine_smiles_series.tolist()
    enamine_row_indices = enamine_smiles_series.index.tolist()  # original row indices
    
    # Apply limit if specified
    max_compounds = config['input'].get('max_compounds', None)
    if max_compounds:
        enamine_smiles = enamine_smiles[:max_compounds]
        enamine_row_indices = enamine_row_indices[:max_compounds]
    
    print(f"✓ Loaded {len(enamine_smiles):,} compounds for screening\n", flush=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Load or compute embeddings
    # ─────────────────────────────────────────────────────────────────────────
    
    print("STAGE 2: Computing/Loading Ligand Embeddings", flush=True)
    
    cache_path = Path(config['embeddings']['cache_path'])
    use_cache = config['embeddings'].get('use_cache', True)
    
    if use_cache and cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path}...", flush=True)
        with open(cache_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        print(f"  ✓ Loaded {len(embeddings_dict):,} cached embeddings", flush=True)
        
        # Check if we need to compute any missing embeddings
        cached_smiles = set(embeddings_dict.keys())
        missing_smiles = [s for s in enamine_smiles if s not in cached_smiles]
        
        if missing_smiles:
            print(f"  Found {len(missing_smiles):,} compounds without cached embeddings", flush=True)
            print(f"  Initializing Uni-Mol encoder (this takes 2-3 min, please wait)...", flush=True)
            with suppress_output():
                from unimol_tools import UniMolRepr
                mol_repr = UniMolRepr(
                    data_type='molecule',
                    remove_hs=True,
                    use_cuda=False,
                    model_name='unimolv1',
                )
            print(f"  ✓ Uni-Mol encoder loaded", flush=True)
            print(f"  ✓ Uni-Mol model ready", flush=True)
            
            print(f"Computing missing embeddings...")
            new_embeddings, failed = compute_embeddings_batch(
                missing_smiles, 
                mol_repr, 
                batch_size=config['embeddings'].get('batch_size', 32),
                verbose=True
            )
            
            embeddings_dict.update(new_embeddings)
            
            # Save updated cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings_dict, f)
            print(f"✓ Updated cache with {len(new_embeddings):,} new embeddings\n")
            
            if failed:
                print(f"⚠ Failed to compute {len(failed)} embeddings")
                failed_path = cache_path.parent / (cache_path.stem + '_failed.txt')
                with open(failed_path, 'w') as f:
                    for smi, reason in failed:
                        f.write(f"{smi}\t{reason}\n")
                print(f"Saved failed SMILES to {failed_path}\n")
    
    else:
        print(f"  No embedding cache found at {cache_path}", flush=True)
        print(f"  Initializing Uni-Mol encoder (this takes 2-3 min, please wait)...", flush=True)
        with suppress_output():
            from unimol_tools import UniMolRepr
            mol_repr = UniMolRepr(
                data_type='molecule',
                remove_hs=True,
                use_cuda=False,
                model_name='unimolv1',
            )
        print(f"  ✓ Uni-Mol encoder loaded", flush=True)
        print(f"  ✓ Uni-Mol model ready", flush=True)
        print(f"  Computing embeddings for all {len(enamine_smiles):,} compounds...", flush=True)
        embeddings_dict, failed = compute_embeddings_batch(
            enamine_smiles,
            mol_repr, 
            batch_size=config['embeddings'].get('batch_size', 32),
            verbose=True
        )
        
        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f"\n✓ Saved {len(embeddings_dict):,} embeddings to {cache_path}\n")
        
        if failed:
            print(f"⚠ Failed to compute {len(failed)} embeddings")
            failed_path = cache_path.parent / (cache_path.stem + '_failed.txt')
            with open(failed_path, 'w') as f:
                for smi, reason in failed:
                    f.write(f"{smi}\t{reason}\n")
            print(f"Saved failed SMILES to {failed_path}\n")
    
    # Filter to only compounds with embeddings
    valid_pairs = [(s, idx) for s, idx in zip(enamine_smiles, enamine_row_indices) if s in embeddings_dict]
    valid_smiles = [s for s, _ in valid_pairs]
    valid_row_indices = [idx for _, idx in valid_pairs]
    print(f"✓ Proceeding with {len(valid_smiles):,} compounds (embeddings available)\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Load pocket embeddings
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"{'─'*70}")
    print("STAGE 3: Loading Pocket Embeddings")
    print(f"{'─'*70}")
    
    pocket_path = Path(config['pockets']['pocket_embeddings_path'])
    print(f"Loading pocket embeddings from {pocket_path}...")
    
    with open(pocket_path, 'rb') as f:
        pocket_data = pickle.load(f)
    
    pocket_names = pocket_data['pocket_names']
    pocket_embeddings = pocket_data['cls_repr']  # (6, 512)
    
    pgk1_indices = [i for i, name in enumerate(pocket_names) if 'PGK1' in name]
    pgk2_indices = [i for i, name in enumerate(pocket_names) if 'PGK2' in name]
    
    pgk1_mean = pocket_embeddings[pgk1_indices].mean(axis=0)  # (512,)
    pgk2_mean = pocket_embeddings[pgk2_indices].mean(axis=0)  # (512,)
    
    print(f"✓ Loaded {len(pocket_names)} pockets")
    print(f"  PGK1_mean: from {len(pgk1_indices)} pockets")
    print(f"  PGK2_mean: from {len(pgk2_indices)} pockets\n")
    
    # Convert to tensors
    pgk1_mean_tensor = torch.tensor(pgk1_mean, dtype=torch.float32, device=device)
    pgk2_mean_tensor = torch.tensor(pgk2_mean, dtype=torch.float32, device=device)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Load Stage 1B model checkpoint
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"{'─'*70}")
    print("STAGE 4: Loading Trained Models")
    print(f"{'─'*70}")
    
    checkpoint_path = Path(config['model']['checkpoint_path'])
    print(f"Loading Stage 1B model checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Auto-detect metric head from checkpoint keys
    _state = checkpoint['model_state_dict']
    _has_metric_head = any('metric_projector' in k for k in _state.keys())

    model_stage1 = SelectivityModel(
        ligand_dim=512,
        pocket_dim=512,
        ligand_hidden_dim=256,
        pocket_proj_dim=128,
        use_metric_head=_has_metric_head,
    ).to(device)

    model_stage1.load_state_dict(_state)
    model_stage1.eval()
    
    print(f"✓ Stage 1B checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"✓ Stage 1B validation AUC: {checkpoint.get('val_auc', 'unknown'):.4f}\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4b: Load Stage 2 LOO ensemble (optional)
    # ─────────────────────────────────────────────────────────────────────────
    
    use_stage2_ensemble = config['model'].get('use_stage2_ensemble', False)
    stage2_models = []
    
    if use_stage2_ensemble:
        print(f"Initializing Stage 2 LOO ensemble...")
        stage2_dir = Path(config['model']['stage2_loo_dir'])
        print(f"Loading {15} Stage 2 fold checkpoints from {stage2_dir}...")
        
        for fold_idx in range(1, 16):
            fold_dir = stage2_dir / f"fold_{fold_idx}"
            fold_ckpt_path = fold_dir / f"fold_{fold_idx}_best.pt"
            
            if not fold_ckpt_path.exists():
                print(f"  WARNING: {fold_ckpt_path} not found, skipping fold {fold_idx}")
                continue
            
            fold_checkpoint = torch.load(fold_ckpt_path, map_location=device)
            _fold_state = fold_checkpoint['model_state_dict']
            _fold_has_metric = any('metric_projector' in k for k in _fold_state.keys())
            fold_model = SelectivityModel(
                ligand_dim=512,
                pocket_dim=512,
                ligand_hidden_dim=256,
                pocket_proj_dim=128,
                use_metric_head=_fold_has_metric,
            ).to(device)
            fold_model.load_state_dict(_fold_state)
            fold_model.eval()
            stage2_models.append(fold_model)
        
        print(f"Loaded {len(stage2_models)}/15 Stage 2 LOO fold models\n")
    else:
        print("Stage 2 ensemble disabled, using only Stage 1B predictions\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Run forward pass on all compounds
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"{'─'*70}")
    print("STAGE 5: Running Forward Pass (Stage 1B Binding Predictions)")
    print(f"{'─'*70}")
    print(f"Screening {len(valid_smiles):,} compounds...\n")
    
    results = []
    batch_size = config['inference'].get('batch_size', 256)
    
    with torch.no_grad():
        for i in range(0, len(valid_smiles), batch_size):
            batch_smiles = valid_smiles[i:i+batch_size]
            screened_so_far = min(i + batch_size, len(valid_smiles))
            if (i // 1000) < (screened_so_far // 1000) or screened_so_far == len(valid_smiles):
                print(f"  Screened {screened_so_far:,} / {len(valid_smiles):,}", flush=True)
            
            # Get ligand embeddings
            ligand_embs = np.array([embeddings_dict[smi] for smi in batch_smiles])
            ligand_tensor = torch.tensor(ligand_embs, dtype=torch.float32, device=device)
            
            # Expand pocket tensors to match batch size
            batch_len = len(batch_smiles)
            pgk1_batch = pgk1_mean_tensor.unsqueeze(0).expand(batch_len, -1)
            pgk2_batch = pgk2_mean_tensor.unsqueeze(0).expand(batch_len, -1)
            
            # STAGE 1B: Forward pass for PGK1 and PGK2
            logits_pgk1, _ = model_stage1(ligand_tensor, pgk1_batch)
            p_bind_pgk1_stage1 = torch.sigmoid(logits_pgk1).squeeze().cpu().numpy()
            
            logits_pgk2, _ = model_stage1(ligand_tensor, pgk2_batch)
            p_bind_pgk2_stage1 = torch.sigmoid(logits_pgk2).squeeze().cpu().numpy()
            
            # Compute Stage 1B selectivity
            if batch_len == 1:
                p_bind_pgk1_stage1 = np.array([p_bind_pgk1_stage1])
                p_bind_pgk2_stage1 = np.array([p_bind_pgk2_stage1])
            
            selectivity_stage1 = p_bind_pgk2_stage1 - p_bind_pgk1_stage1
            
            # Store Stage 1B results
            for j, smi in enumerate(batch_smiles):
                results.append({
                    'smiles': smi,
                    'p_bind_pgk1_stage1': float(p_bind_pgk1_stage1[j]),
                    'p_bind_pgk2_stage1': float(p_bind_pgk2_stage1[j]),
                    'selectivity_stage1': float(selectivity_stage1[j]),
                })
        
        # STAGE 2: LOO Ensemble predictions (if enabled)
        if use_stage2_ensemble and len(stage2_models) > 0:
            print(f"\n{'─'*70}")
            print(f"STAGE 6: Running Stage 2 Ensemble ({len(stage2_models)} LOO Folds)")
            print(f"{'─'*70}\n")
            
            for i in range(0, len(valid_smiles), batch_size):
                batch_smiles = valid_smiles[i:i+batch_size]
                ensemble_so_far = min(i + batch_size, len(valid_smiles))
                if (i // 1000) < (ensemble_so_far // 1000) or ensemble_so_far == len(valid_smiles):
                    print(f"  Ensemble voted {ensemble_so_far:,} / {len(valid_smiles):,}", flush=True)
                
                # Get ligand embeddings
                ligand_embs = np.array([embeddings_dict[smi] for smi in batch_smiles])
                ligand_tensor = torch.tensor(ligand_embs, dtype=torch.float32, device=device)
                
                # Expand pocket tensors
                batch_len = len(batch_smiles)
                pgk1_batch = pgk1_mean_tensor.unsqueeze(0).expand(batch_len, -1)
                pgk2_batch = pgk2_mean_tensor.unsqueeze(0).expand(batch_len, -1)
                
                # Collect predictions from all folds
                fold_selectivities = []  # shape: (n_folds, batch_len)
                
                for fold_model in stage2_models:
                    logits_pgk1, _ = fold_model(ligand_tensor, pgk1_batch)
                    p_pgk1 = torch.sigmoid(logits_pgk1).squeeze().cpu().numpy()
                    
                    logits_pgk2, _ = fold_model(ligand_tensor, pgk2_batch)
                    p_pgk2 = torch.sigmoid(logits_pgk2).squeeze().cpu().numpy()
                    
                    if batch_len == 1:
                        p_pgk1 = np.array([p_pgk1])
                        p_pgk2 = np.array([p_pgk2])
                    
                    selectivity_fold = p_pgk2 - p_pgk1
                    fold_selectivities.append(selectivity_fold)
                
                fold_selectivities = np.array(fold_selectivities)  # (n_folds, batch_len)
                
                # Compute ensemble statistics
                mean_selectivity = fold_selectivities.mean(axis=0)
                std_selectivity = fold_selectivities.std(axis=0)
                votes_pgk2 = (fold_selectivities > 0).sum(axis=0)  # count positive selectivity
                vote_fraction = votes_pgk2 / len(stage2_models)
                confidence_score = vote_fraction * np.abs(mean_selectivity)
                
                # Update results with Stage 2 metrics
                for j, smi in enumerate(batch_smiles):
                    result_idx = i + j
                    results[result_idx].update({
                        'selectivity_stage2_mean': float(mean_selectivity[j]),
                        'selectivity_stage2_std': float(std_selectivity[j]),
                        'votes_pgk2_selective': int(votes_pgk2[j]),
                        'vote_fraction': float(vote_fraction[j]),
                        'confidence_score': float(confidence_score[j]),
                    })
    
    print(f"\n{'─'*70}")
    print("STAGE 7: Saving Results")
    print(f"{'─'*70}\n")
    
    print(f"✓ Completed predictions for {len(results):,} compounds\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Rank and save results
    # ─────────────────────────────────────────────────────────────────────────
    
    df_results = pd.DataFrame(results)
    df_results['_row_idx'] = valid_row_indices
    
    # Merge with original metadata 1-to-1 by original row index (avoids SMILES-duplicate issue)
    df_meta = df_enamine.drop(columns=[smiles_col], errors='ignore')  # smiles already in df_results
    df_results = df_results.merge(df_meta, left_on='_row_idx', right_index=True, how='left')
    df_results = df_results.drop(columns=['_row_idx'])
    
    # Determine ranking strategy
    if use_stage2_ensemble and len(stage2_models) > 0:
        print(f"Ranking by Stage 2 ensemble confidence...")
        
        # Primary sort: votes_pgk2_selective (descending)
        # Secondary sort: |selectivity_stage2_mean| (descending)
        # Tertiary sort: -selectivity_stage2_std (ascending, i.e., lower std first)
        df_results['_abs_sel'] = df_results['selectivity_stage2_mean'].abs()
        df_results = df_results.sort_values(
            by=['votes_pgk2_selective', '_abs_sel', 'selectivity_stage2_std'],
            ascending=[False, False, True]
        ).drop(columns=['_abs_sel'])
        
        # Categorize selectivity based on votes
        def categorize_selectivity(row):
            votes = row['votes_pgk2_selective']
            n_folds = len(stage2_models)
            if votes >= n_folds * 0.8:  # 80%+ vote for PGK2
                return 'PGK2-selective (high confidence)'
            elif votes >= n_folds * 0.6:  # 60-80%
                return 'PGK2-selective (moderate confidence)'
            elif votes >= n_folds * 0.4:  # 40-60%
                return 'Uncertain'
            elif votes >= n_folds * 0.2:  # 20-40%
                return 'PGK1-selective (moderate confidence)'
            else:  # <20%
                return 'PGK1-selective (high confidence)'
        
        df_results['selectivity_category'] = df_results.apply(categorize_selectivity, axis=1)
        
    else:
        if config['output'].get('preserve_order', False):
            print(f"Preserving original compound order (preserve_order=true)...")
        else:
            print(f"Ranking by Stage 1B selectivity score...")
            # Sort by Stage 1B selectivity score (most PGK2-selective first)
            df_results = df_results.sort_values('selectivity_stage1', ascending=False)

        # Add plain selectivity label
        df_results['selectivity'] = df_results['selectivity_stage1'].apply(
            lambda s: 'PGK2' if s > 0 else 'PGK1'
        )
    
    # Save full results
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    full_results_path = output_dir / config['output']['full_results_filename']
    df_results.to_csv(full_results_path, index=False)
    print(f"✓ Saved full results for all {len(df_results):,} compounds → {full_results_path}\n")
    
    if use_stage2_ensemble and len(stage2_models) > 0:
        # All PGK2-selective (votes > 7, i.e. majority)
        df_pgk2 = df_results[df_results['votes_pgk2_selective'] > len(stage2_models) / 2]
        pgk2_path = output_dir / config['output']['pgk2_selective_filename']
        df_pgk2.to_csv(pgk2_path, index=False)
        print(f"✓ Saved {len(df_pgk2):,} PGK2-selective compounds → {pgk2_path}", flush=True)
        
        # All PGK1-selective (votes < 7, i.e. majority vote against PGK2)
        df_pgk1 = df_results[df_results['votes_pgk2_selective'] < len(stage2_models) / 2].iloc[::-1]
        pgk1_path = output_dir / config['output']['pgk1_selective_filename']
        df_pgk1.to_csv(pgk1_path, index=False)
        print(f"✓ Saved {len(df_pgk1):,} PGK1-selective compounds → {pgk1_path}", flush=True)
        
    else:
        # All PGK2-selective
        df_pgk2 = df_results[df_results['selectivity_stage1'] > 0]
        pgk2_path = output_dir / config['output']['pgk2_selective_filename']
        df_pgk2.to_csv(pgk2_path, index=False)
        print(f"✓ Saved {len(df_pgk2):,} PGK2-selective compounds → {pgk2_path}", flush=True)
        
        # All PGK1-selective
        df_pgk1 = df_results[df_results['selectivity_stage1'] < 0]
        pgk1_path = output_dir / config['output']['pgk1_selective_filename']
        df_pgk1.to_csv(pgk1_path, index=False)
        print(f"✓ Saved {len(df_pgk1):,} PGK1-selective compounds → {pgk1_path}", flush=True)
    
    # Print summary statistics
    print(f"\n{'─'*70}")
    print("SUMMARY STATISTICS")
    print(f"{'─'*70}")
    print(f"Total compounds screened: {len(df_results):,}")
    
    if use_stage2_ensemble and len(stage2_models) > 0:
        print(f"\nStage 2 Ensemble Statistics ({len(stage2_models)} folds):")
        print(f"  Mean selectivity: {df_results['selectivity_stage2_mean'].mean():.4f}")
        print(f"  Std selectivity:  {df_results['selectivity_stage2_mean'].std():.4f}")
        print(f"  Min selectivity:  {df_results['selectivity_stage2_mean'].min():.4f}")
        print(f"  Max selectivity:  {df_results['selectivity_stage2_mean'].max():.4f}")
        print(f"\nVote distribution:")
        vote_counts = df_results['votes_pgk2_selective'].value_counts().sort_index(ascending=False)
        for votes, count in vote_counts.items():
            print(f"  {votes:2d} votes: {count:5d} compounds ({count/len(df_results)*100:.1f}%)")
        print(f"\nSelectivity categories:")
        cat_counts = df_results['selectivity_category'].value_counts()
        for cat, count in cat_counts.items():
            print(f"  {cat}: {count:,} ({count/len(df_results)*100:.1f}%)")
        print(f"\nTop 5 highest confidence PGK2-selective:")
        display_cols = ['smiles', 'votes_pgk2_selective', 'selectivity_stage2_mean', 'selectivity_stage2_std', 'confidence_score']
        print(df_results[display_cols].head(5).to_string(index=False))
    else:
        print(f"\nStage 1B Selectivity distribution:")
        print(f"  Mean: {df_results['selectivity_stage1'].mean():.4f}")
        print(f"  Std:  {df_results['selectivity_stage1'].std():.4f}")
        print(f"  Min:  {df_results['selectivity_stage1'].min():.4f}")
        print(f"  Max:  {df_results['selectivity_stage1'].max():.4f}")
        print(f"\nTop 5 PGK2-selective compounds:")
        display_cols = ['smiles', 'selectivity_stage1', 'p_bind_pgk1_stage1', 'p_bind_pgk2_stage1']
        print(df_results[display_cols].head(5).to_string(index=False))
        print(f"\nTop 5 PGK1-selective compounds:")
        print(df_results[display_cols].tail(5).to_string(index=False))
    
    print(f"{'─'*70}\n")
    
    print("✓ SCREENING COMPLETE!\n", flush=True)
    print(f"Results saved to: {output_dir}/", flush=True)
    print(f"  - Full results: all {len(df_results):,} compounds", flush=True)
    print(f"  - PGK2-selective: {len(df_pgk2):,} compounds", flush=True)
    print(f"  - PGK1-selective: {len(df_pgk1):,} compounds\n", flush=True)


if __name__ == '__main__':
    print("[screen_enamine] Parsing arguments...", flush=True)
    parser = argparse.ArgumentParser(description='Screen Enamine library for PGK selectivity')
    parser.add_argument('--config', type=str, default='config_screen_enamine.yaml',
                       help='Path to config YAML file')
    args = parser.parse_args()
    print(f"[screen_enamine] Config: {args.config}", flush=True)
    screen_compounds(config_path=args.config)
