"""
Parallelized version of screen_enamine.py for ComputeCanada/GPU clusters.

Combines:
  - Multiprocessing (CPU cores) for RDKit conformer generation
  - GPU batching for Uni-Mol embeddings (batch_size=256)
  - Chunk-based processing (20k compounds per chunk)

Pipeline:
  1. Split SMILES into chunks (20k each)
  2. For each chunk:
     a. Parallelize conformer generation across CPU cores
     b. GPU-batch embeddings (256/batch)
     c. Save chunk cache
  3. Merge all chunk caches
  4. Run normal screening on merged cache

Usage:
  conda activate unimol
  python parallelized_screen_enamine.py --config config_screen_enamine_v1.yaml --num_workers 8
"""

import sys
print("[parallelized_screen] Python started, loading imports...", flush=True)

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
from multiprocessing import Pool, cpu_count

print("[parallelized_screen] Loading torch...", flush=True)
import torch
print("[parallelized_screen] Loading RDKit...", flush=True)
from rdkit import Chem
from rdkit.Chem import AllChem

# Suppress verbose logging from unimol_tools
logging.getLogger('unimol_tools').setLevel(logging.ERROR)
logging.getLogger('unimol_tools.tasks').setLevel(logging.ERROR)
logging.getLogger('unimol_tools.tasks.trainer').setLevel(logging.ERROR)

print("[parallelized_screen] Loading model...", flush=True)
from model import SelectivityModel
print("[parallelized_screen] All imports done. Starting...", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Output suppression
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def suppress_output():
    """Suppress stdout, stderr, and logging temporarily."""
    log_level = logging.root.level
    logging.root.setLevel(logging.CRITICAL)
    for logger_name in ['unimol_tools', 'unimol_tools.tasks', 'unimol_tools.tasks.trainer']:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    
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
            logging.root.setLevel(log_level)
            for logger_name in ['unimol_tools', 'unimol_tools.tasks', 'unimol_tools.tasks.trainer']:
                logging.getLogger(logger_name).setLevel(logging.ERROR)


# ─────────────────────────────────────────────────────────────────────────────
# Conformer generation (RDKit — CPU parallel)
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


def generate_conformer_worker(smiles):
    """
    Worker function for multiprocessing: conformer generation only (no UniMol).
    Returns (smiles, atoms, coords) or (smiles, None, reason) if failed.
    CPU-only, fully picklable.
    """
    try:
        mol = smiles_to_conformer(smiles)
        if mol is None:
            return (smiles, None, None, "conformer_failed")
        
        atoms, coords = mol_to_unimol_input(mol)
        if atoms is None or coords is None:
            return (smiles, None, None, "atomization_failed")
        
        return (smiles, atoms, coords, None)
    except Exception as e:
        return (smiles, None, None, f"error: {str(e)[:50]}")


def compute_embeddings_batch_parallel(smiles_list, mol_repr, num_workers=8, embed_batch_size=256, verbose=True):
    """
    Compute embeddings for a list of SMILES using:
      - Multiprocessing (CPU) for RDKit conformer generation
      - GPU batching for UniMol embeddings
    
    Parameters
    ----------
    smiles_list      : list of str
    mol_repr         : UniMolRepr instance
    num_workers      : int  — CPU workers for conformer generation
    embed_batch_size : int  — UniMol batch size for GPU embedding
    verbose          : bool
    
    Returns
    -------
    embeddings_dict : dict {smiles: embedding (512,)}
    failed_smiles   : list of (smiles, reason)
    """
    embeddings_dict = {}
    failed_smiles = []
    
    # STEP 1: Parallel conformer generation (CPU, picklable)
    if verbose:
        print(f"  Generating 3D conformers ({num_workers} CPU workers)...", flush=True)
    
    conformer_data = []  # list of (smiles, atoms, coords)
    
    with Pool(num_workers) as pool:
        results = pool.imap_unordered(generate_conformer_worker, smiles_list, chunksize=200)
        for smiles, atoms, coords, error in results:
            if error:
                failed_smiles.append((smiles, error))
            else:
                conformer_data.append((smiles, atoms, coords))
    
    if verbose:
        print(f"  ✓ {len(conformer_data):,} conformers ready ({len(failed_smiles)} failed)", flush=True)
        print(f"  Embedding on GPU (batch size: {embed_batch_size})...", flush=True)
    
    # STEP 2: GPU batched embedding — fall back to one-at-a-time if batch fails
    _first_error_printed = False
    for batch_start in range(0, len(conformer_data), embed_batch_size):
        batch = conformer_data[batch_start:batch_start + embed_batch_size]
        batch_smiles = [item[0] for item in batch]
        batch_atoms  = [item[1] for item in batch]
        # Convert coords to list-of-lists (some UniMol versions require this)
        batch_coords = [item[2].tolist() if hasattr(item[2], 'tolist') else item[2] for item in batch]
        
        batch_ok = False
        # Try batched first
        try:
            with suppress_output():
                result = mol_repr.get_repr({'atoms': batch_atoms, 'coordinates': batch_coords})
            embeddings = np.array(result)
            if not _first_error_printed:
                print(f"  DEBUG result type={type(result)}, embeddings.shape={embeddings.shape}, expected=({len(batch_smiles)}, 512)", flush=True)
                _first_error_printed = True
            if embeddings.ndim == 2 and embeddings.shape[0] == len(batch_smiles):
                for smi, emb in zip(batch_smiles, embeddings):
                    embeddings_dict[smi] = emb
                batch_ok = True
        except Exception as e:
            import traceback as _tb
            print(f"  BATCH EMBED EXCEPTION: {e}", flush=True)
            print(_tb.format_exc(), flush=True)

        # Fall back to one molecule at a time (same as original screen_enamine.py)
        if not batch_ok:
            for smi, atoms, coords in zip(batch_smiles, batch_atoms, batch_coords):
                try:
                    with suppress_output():
                        result = mol_repr.get_repr({'atoms': [atoms], 'coordinates': [coords]})
                    emb = np.array(result)[0]
                    embeddings_dict[smi] = emb
                except Exception as e2:
                    failed_smiles.append((smi, f"embed_error: {e2}"))
    
    return embeddings_dict, failed_smiles


# ─────────────────────────────────────────────────────────────────────────────
# Chunk-based screening pipeline
# ─────────────────────────────────────────────────────────────────────────────

def screen_compounds_parallel(config_path='config_screen_enamine_v1.yaml', num_workers=8, chunk_size=20000):
    """
    Parallelized screening pipeline.
    """
    print("\n" + "="*70, flush=True)
    print("PARALLELIZED SCREENING: Enamine Library for PGK1/PGK2 Selectivity", flush=True)
    print("="*70 + "\n", flush=True)
    
    # Load config
    print(f"Loading config from {config_path}...", flush=True)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded\n", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"CPU workers: {num_workers}")
    print(f"Chunk size: {chunk_size:,} compounds\n", flush=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Load Enamine CSV
    # ─────────────────────────────────────────────────────────────────────────
    
    print("STAGE 1: Loading Enamine Compound Library", flush=True)
    
    enamine_csv = Path(config['input']['enamine_csv'])
    print(f"  Loading Enamine compounds from {enamine_csv}...", flush=True)
    df_enamine = pd.read_csv(enamine_csv)
    
    smiles_col = config['input'].get('smiles_column', 'smiles')
    df_enamine = df_enamine.reset_index(drop=True)
    enamine_smiles_series = df_enamine[smiles_col].dropna()
    enamine_smiles = enamine_smiles_series.tolist()
    enamine_row_indices = enamine_smiles_series.index.tolist()
    
    max_compounds = config['input'].get('max_compounds', None)
    if max_compounds:
        enamine_smiles = enamine_smiles[:max_compounds]
        enamine_row_indices = enamine_row_indices[:max_compounds]
    
    print(f"✓ Loaded {len(enamine_smiles):,} compounds\n", flush=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Parallel embedding computation with chunking
    # ─────────────────────────────────────────────────────────────────────────
    
    print("STAGE 2: Computing/Loading Ligand Embeddings (Parallel)", flush=True)
    
    cache_path = Path(config['embeddings']['cache_path'])
    use_cache = config['embeddings'].get('use_cache', True)
    
    embeddings_dict = {}
    
    # Load existing cache if available
    if use_cache and cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path}...", flush=True)
        with open(cache_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        print(f"  ✓ Loaded {len(embeddings_dict):,} cached embeddings", flush=True)
        # Sanity check: verify cache keys look like SMILES and values are right shape
        sample_keys = list(embeddings_dict.keys())[:3]
        sample_vals = [embeddings_dict[k] for k in sample_keys]
        print(f"  Cache sample keys: {sample_keys}", flush=True)
        print(f"  Cache sample value shapes: {[np.array(v).shape for v in sample_vals]}", flush=True)
        # Check overlap with current SMILES
        overlap = sum(1 for s in enamine_smiles if s in embeddings_dict)
        print(f"  Cache overlap with input SMILES: {overlap:,} / {len(enamine_smiles):,}", flush=True)
        if overlap == 0:
            print(f"  WARNING: Cache has 0 overlap with input SMILES — cache may be stale or from a different file. Recomputing.", flush=True)
            embeddings_dict = {}
        print("", flush=True)
    
    # Check for missing embeddings
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
                use_cuda=(device.type == 'cuda'),
                model_name='unimolv1',
            )
        print(f"  ✓ Uni-Mol encoder loaded\n", flush=True)
        
        # Process in chunks
        num_chunks = (len(missing_smiles) + chunk_size - 1) // chunk_size
        print(f"  Processing {len(missing_smiles):,} compounds in {num_chunks} chunks...\n")
        
        failed_all = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(missing_smiles))
            chunk_smiles = missing_smiles[start_idx:end_idx]
            
            print(f"  Chunk {chunk_idx + 1}/{num_chunks}: {len(chunk_smiles):,} compounds")
            
            # Parallel conformer gen + GPU batched embedding
            new_embeddings, failed = compute_embeddings_batch_parallel(
                chunk_smiles,
                mol_repr,
                num_workers=num_workers,
                embed_batch_size=256,
                verbose=True
            )
            
            embeddings_dict.update(new_embeddings)
            failed_all.extend(failed)
            
            print(f"    → Computed {len(new_embeddings):,} embeddings (failed: {len(failed)})")
        
        # Save merged cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f"\n✓ Saved {len(embeddings_dict):,} embeddings to {cache_path}\n")
        
        if failed_all:
            print(f"⚠ Failed to compute {len(failed_all)} embeddings")
            failed_path = cache_path.parent / (cache_path.stem + '_failed.txt')
            with open(failed_path, 'w') as f:
                for smi, reason in failed_all:
                    f.write(f"{smi}\t{reason}\n")
            print(f"Saved failed SMILES to {failed_path}\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Screening (same as original)
    # ─────────────────────────────────────────────────────────────────────────
    
    print("STAGE 3+: Running Model Inference (GPU batched)")
    print("(Using same screening logic as screen_enamine.py)")
    print()
    
    # Load model, pockets, and run screening
    # (This is the same as the original screen_enamine.py from this point on)
    
    valid_smiles = [s for s in enamine_smiles if s in embeddings_dict]
    print(f"✓ embeddings_dict size: {len(embeddings_dict):,}", flush=True)
    print(f"✓ enamine_smiles size:  {len(enamine_smiles):,}", flush=True)
    print(f"✓ valid_smiles (overlap): {len(valid_smiles):,}", flush=True)
    if len(valid_smiles) == 0:
        sample_input = enamine_smiles[:3]
        sample_cache = list(embeddings_dict.keys())[:3]
        print(f"  SMILES sample from input:  {sample_input}", flush=True)
        print(f"  SMILES sample from cache:  {sample_cache}", flush=True)
        raise RuntimeError("valid_smiles is empty — 0 overlap between input SMILES and embeddings_dict. Cache mismatch or wrong smiles_column.")
    print(f"✓ Proceeding with {len(valid_smiles):,} compounds (embeddings available)\n", flush=True)
    
    # Load model (same logic as screen_enamine.py — auto-detect metric head)
    print("Loading Stage 1 model...", flush=True)
    model_path = Path(config['model']['checkpoint_path'])
    checkpoint = torch.load(model_path, map_location=device)

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
    print(f"✓ Model loaded from {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val AUC: {checkpoint.get('val_auc', 'unknown')}\n", flush=True)

    # Load pocket embeddings (same logic as screen_enamine.py — filter by pocket_names)
    print("Loading pocket embeddings...", flush=True)
    pocket_path = Path(config['pockets']['pocket_embeddings_path'])
    with open(pocket_path, 'rb') as f:
        pocket_data = pickle.load(f)

    pocket_names = pocket_data['pocket_names']
    pocket_embeddings = pocket_data['cls_repr']  # (N, 512)

    pgk1_indices = [i for i, name in enumerate(pocket_names) if 'PGK1' in name]
    pgk2_indices = [i for i, name in enumerate(pocket_names) if 'PGK2' in name]

    pgk1_mean = pocket_embeddings[pgk1_indices].mean(axis=0)
    pgk2_mean = pocket_embeddings[pgk2_indices].mean(axis=0)

    pgk1_mean_tensor = torch.tensor(pgk1_mean, dtype=torch.float32, device=device)
    pgk2_mean_tensor = torch.tensor(pgk2_mean, dtype=torch.float32, device=device)
    print(f"✓ Pocket embeddings loaded")
    print(f"  PGK1_mean: from {len(pgk1_indices)} pockets")
    print(f"  PGK2_mean: from {len(pgk2_indices)} pockets\n", flush=True)
    
    # Run screening
    results = []
    batch_size = config['inference'].get('batch_size', 256)
    
    print(f"Screening {len(valid_smiles):,} compounds (batch size: {batch_size})...\n", flush=True)
    
    with torch.no_grad():
        for i in range(0, len(valid_smiles), batch_size):
            batch_smiles = valid_smiles[i:i+batch_size]
            screened_so_far = min(i + batch_size, len(valid_smiles))
            if (i // 10000) < (screened_so_far // 10000) or screened_so_far == len(valid_smiles):
                print(f"  Screened {screened_so_far:,} / {len(valid_smiles):,}", flush=True)
            
            ligand_embs = np.array([embeddings_dict[smi] for smi in batch_smiles])
            ligand_tensor = torch.tensor(ligand_embs, dtype=torch.float32, device=device)
            
            batch_len = len(batch_smiles)
            pgk1_batch = pgk1_mean_tensor.unsqueeze(0).expand(batch_len, -1)
            pgk2_batch = pgk2_mean_tensor.unsqueeze(0).expand(batch_len, -1)
            
            logits_pgk1, _ = model_stage1(ligand_tensor, pgk1_batch)
            p_bind_pgk1_stage1 = torch.sigmoid(logits_pgk1).squeeze().cpu().numpy()
            
            logits_pgk2, _ = model_stage1(ligand_tensor, pgk2_batch)
            p_bind_pgk2_stage1 = torch.sigmoid(logits_pgk2).squeeze().cpu().numpy()
            
            if batch_len == 1:
                p_bind_pgk1_stage1 = np.array([p_bind_pgk1_stage1])
                p_bind_pgk2_stage1 = np.array([p_bind_pgk2_stage1])
            
            selectivity_stage1 = p_bind_pgk2_stage1 - p_bind_pgk1_stage1
            
            for j, smi in enumerate(batch_smiles):
                results.append({
                    'smiles': smi,
                    'p_bind_pgk1_stage1': float(p_bind_pgk1_stage1[j]),
                    'p_bind_pgk2_stage1': float(p_bind_pgk2_stage1[j]),
                    'selectivity_stage1': float(selectivity_stage1[j]),
                })
    
    print(f"\n✓ Screening complete. Processing {len(results):,} results...\n", flush=True)
    
    # Save results
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('selectivity_stage1', ascending=False)
    
    full_results_path = output_dir / config['output']['full_results_filename']
    results_df.to_csv(full_results_path, index=False)
    print(f"✓ Saved full results to {full_results_path}")
    
    # Save top PGK2-selective
    pgk2_top = results_df.head(1000)
    pgk2_path = output_dir / config['output']['pgk2_selective_filename']
    pgk2_top.to_csv(pgk2_path, index=False)
    print(f"✓ Saved top 1000 PGK2-selective to {pgk2_path}")
    
    # Save top PGK1-selective
    pgk1_top = results_df.tail(1000)
    pgk1_path = output_dir / config['output']['pgk1_selective_filename']
    pgk1_top.to_csv(pgk1_path, index=False)
    print(f"✓ Saved top 1000 PGK1-selective to {pgk1_path}\n")
    
    print("="*70)
    print("SCREENING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parallelized screening with GPU batching')
    parser.add_argument('--config', default='config_screen_enamine_v1.yaml', help='Config file path')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of CPU workers for conformer generation')
    parser.add_argument('--chunk_size', type=int, default=20000, help='Compounds per chunk')
    args = parser.parse_args()
    
    screen_compounds_parallel(
        config_path=args.config,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size
    )
