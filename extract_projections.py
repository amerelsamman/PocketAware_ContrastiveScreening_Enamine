"""
Extract internal model projections for downstream analysis (PCA, t-SNE, Tanimoto).

For each compound this script saves:
  - ligand_repr   (256-dim)  pre-FiLM ligand encoding — pure chemistry, pocket-blind
  - metric_emb    ( 64-dim)  L2-normalized metric learning space (if checkpoint has head)
  - post_film     (256-dim)  FiLM-modulated repr (run once per pocket: PGK1 and PGK2)
  - morgan_fp     (2048-bit) ECFP4 fingerprint for Tanimoto similarity
  - p_bind_pgk1, p_bind_pgk2, selectivity  (scalar predictions)

Outputs (written to --output_dir):
  projections.npz     — numpy arrays, keys listed above
  projections_meta.csv — SMILES + scalar columns (easy merge by index)

Usage
-----
    python extract_projections.py --config config_diversity_screen.yaml
    python extract_projections.py --config config_diversity_screen.yaml --output_dir my_analysis/
    # Limit to top/bottom N for quick analysis of interesting compounds:
    python extract_projections.py --config config_diversity_screen.yaml --from_results data/.../results.csv --top_n 2000
"""

import sys
print("[extract_projections] Python started, loading imports...", flush=True)

import os
import pickle
import argparse
import yaml
import contextlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

print("[extract_projections] Loading torch...", flush=True)
import torch

print("[extract_projections] Loading RDKit...", flush=True)
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

logging.getLogger('unimol_tools').setLevel(logging.ERROR)

print("[extract_projections] Loading model...", flush=True)
from model import SelectivityModel
print("[extract_projections] Imports done.\n", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (reused from screen_enamine.py)
# ──────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def suppress_output():
    log_level = logging.root.level
    logging.root.setLevel(logging.CRITICAL)
    for name in ['unimol_tools', 'unimol_tools.tasks', 'unimol_tools.tasks.trainer']:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            logging.root.setLevel(log_level)
            for name in ['unimol_tools', 'unimol_tools.tasks', 'unimol_tools.tasks.trainer']:
                logging.getLogger(name).setLevel(logging.ERROR)


def smiles_to_conformer(smiles):
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
        except Exception:
            pass
        return mol
    except Exception:
        return None


def mol_to_unimol_input(mol):
    if mol.GetNumConformers() == 0:
        return None, None
    conf = mol.GetConformer()
    atoms, coords = [], []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        atoms.append(atom.GetSymbol())
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    if not atoms:
        return None, None
    return atoms, np.array(coords, dtype=np.float32)


def compute_morgan_fp(smiles, radius=2, n_bits=2048):
    """Return ECFP4 fingerprint as uint8 bit-array, or None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp  = gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


def ensure_embeddings(smiles_list, cache_path, config_emb):
    """Load / compute Uni-Mol ligand embeddings, return dict {smiles: (512,)}."""
    cache_path = Path(cache_path)
    use_cache  = config_emb.get('use_cache', True)

    embeddings_dict = {}
    if use_cache and cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path}...", flush=True)
        with open(cache_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        print(f"  ✓ {len(embeddings_dict):,} cached embeddings found", flush=True)

    missing = [s for s in smiles_list if s not in embeddings_dict]
    if missing:
        print(f"  Computing {len(missing):,} missing embeddings via Uni-Mol...", flush=True)
        with suppress_output():
            from unimol_tools import UniMolRepr
            mol_repr = UniMolRepr(
                data_type='molecule', remove_hs=True, use_cuda=False, model_name='unimolv1'
            )
        for idx, smi in enumerate(missing):
            try:
                mol = smiles_to_conformer(smi)
                if mol is None:
                    continue
                atoms, coords = mol_to_unimol_input(mol)
                if atoms is None:
                    continue
                with suppress_output():
                    result = mol_repr.get_repr({'atoms': [atoms], 'coordinates': [coords]})
                embeddings_dict[smi] = np.array(result)[0]
            except Exception:
                pass
            if (idx + 1) % 500 == 0:
                print(f"    Embedded {idx+1:,}/{len(missing):,}", flush=True)
        # Update cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f"  ✓ Cache updated", flush=True)

    return embeddings_dict


# ──────────────────────────────────────────────────────────────────────────────
# Main extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract(config_path: str,
            output_dir: str | None = None,
            from_results: str | None = None,
            top_n: int | None = None):
    """
    Parameters
    ----------
    config_path  : path to a screen_enamine YAML config (provides checkpoint, pockets, embeddings cache)
    output_dir   : override the output directory from config
    from_results : (optional) path to an existing screening CSV — skip model re-run and only
                   extract projections for the compounds in that file.  Useful for focusing on
                   top/bottom hits without re-screening the entire library.
    top_n        : if from_results is given, only process the first top_n rows (already sorted
                   by selectivity in screening outputs).
    """
    print("=" * 70, flush=True)
    print("EXTRACT PROJECTIONS for PCA / t-SNE / Tanimoto analysis", flush=True)
    print("=" * 70 + "\n", flush=True)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n", flush=True)

    # ── 1. Determine SMILES list ──────────────────────────────────────────────
    if from_results is not None:
        print(f"Loading SMILES from existing results: {from_results}", flush=True)
        df_src = pd.read_csv(from_results)
        if top_n:
            df_src = df_src.head(top_n)
            print(f"  Using top {top_n} rows", flush=True)
        smiles_col = config['input'].get('smiles_column', 'smiles')
        if smiles_col not in df_src.columns:
            smiles_col = 'smiles'  # fallback
        smiles_list = df_src[smiles_col].dropna().tolist()
        print(f"  ✓ {len(smiles_list):,} compounds to process\n", flush=True)
    else:
        print("Loading SMILES from config input...", flush=True)
        df_src = pd.read_csv(config['input']['enamine_csv'])
        smiles_col = config['input'].get('smiles_column', 'smiles')
        smiles_list = df_src[smiles_col].dropna().tolist()
        max_compounds = config['input'].get('max_compounds', None)
        if max_compounds:
            smiles_list = smiles_list[:max_compounds]
        if top_n:
            smiles_list = smiles_list[:top_n]
        print(f"  ✓ {len(smiles_list):,} compounds\n", flush=True)

    # ── 2. Load / compute ligand embeddings ───────────────────────────────────
    print("─" * 70, flush=True)
    print("Loading ligand embeddings...", flush=True)
    embeddings_dict = ensure_embeddings(
        smiles_list,
        cache_path=config['embeddings']['cache_path'],
        config_emb=config['embeddings'],
    )
    valid_smiles = [s for s in smiles_list if s in embeddings_dict]
    print(f"✓ {len(valid_smiles):,} / {len(smiles_list):,} compounds have embeddings\n", flush=True)

    # ── 3. Load pocket embeddings ─────────────────────────────────────────────
    print("─" * 70, flush=True)
    print("Loading pocket embeddings...", flush=True)
    with open(config['pockets']['pocket_embeddings_path'], 'rb') as f:
        pocket_data = pickle.load(f)

    pocket_names = pocket_data['pocket_names']
    pocket_embs  = pocket_data['cls_repr']

    pgk1_idx = [i for i, n in enumerate(pocket_names) if 'PGK1' in n]
    pgk2_idx = [i for i, n in enumerate(pocket_names) if 'PGK2' in n]
    pgk1_mean_t = torch.tensor(pocket_embs[pgk1_idx].mean(0), dtype=torch.float32, device=device)
    pgk2_mean_t = torch.tensor(pocket_embs[pgk2_idx].mean(0), dtype=torch.float32, device=device)
    print(f"  PGK1_mean from {len(pgk1_idx)} pockets, PGK2_mean from {len(pgk2_idx)} pockets\n", flush=True)

    # ── 4. Load model ─────────────────────────────────────────────────────────
    print("─" * 70, flush=True)
    checkpoint_path = config['model']['checkpoint_path']
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint['model_state_dict']
    has_metric = any('metric_projector' in k for k in state)

    # Detect metric branch from checkpoint if stored, else default post_film
    metric_branch = checkpoint.get('metric_branch', 'post_film')

    model = SelectivityModel(
        ligand_dim=512, pocket_dim=512,
        ligand_hidden_dim=256, pocket_proj_dim=128,
        use_metric_head=has_metric,
        metric_branch=metric_branch,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"  ✓ Loaded (epoch {checkpoint.get('epoch','?')}, "
          f"val_auc {checkpoint.get('val_auc', 0):.4f})", flush=True)
    print(f"  Metric head: {has_metric}  |  Branch: {metric_branch}\n", flush=True)

    # ── 5. Register forward hook to capture pre/post-FiLM representations ─────
    #
    # FiLMLayer.forward(pocket_proj, ligand_repr) → modulated
    #   input[0]  = pocket_proj  (128-dim) — not needed here
    #   input[1]  = ligand_repr  (256-dim) pre-FiLM  ← capture
    #   output    = modulated    (256-dim) post-FiLM  ← capture
    #
    _hook_store = {}

    def _film_hook(module, inputs, output):
        _hook_store['pre_film']  = inputs[1].detach().cpu()   # ligand_repr
        _hook_store['post_film'] = output.detach().cpu()       # modulated

    hook_handle = model.film_layer.register_forward_hook(_film_hook)

    # ── 6. Run batched inference ───────────────────────────────────────────────
    print("─" * 70, flush=True)
    print("Running batched inference and extracting projections...", flush=True)

    batch_size = config['inference'].get('batch_size', 256)
    N = len(valid_smiles)

    # Pre-allocate output arrays
    ligand_repr_all  = np.zeros((N, 256), dtype=np.float32)
    metric_emb_all   = np.zeros((N,  64), dtype=np.float32) if has_metric else None
    post_film_pgk1   = np.zeros((N, 256), dtype=np.float32)   # FiLM output with PGK1 pocket
    post_film_pgk2   = np.zeros((N, 256), dtype=np.float32)   # FiLM output with PGK2 pocket
    p_bind_pgk1_all  = np.zeros(N, dtype=np.float32)
    p_bind_pgk2_all  = np.zeros(N, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = valid_smiles[i : i + batch_size]
            blen  = len(batch)

            lig_embs  = np.stack([embeddings_dict[s] for s in batch])
            lig_t     = torch.tensor(lig_embs, dtype=torch.float32, device=device)
            pgk1_b    = pgk1_mean_t.unsqueeze(0).expand(blen, -1)
            pgk2_b    = pgk2_mean_t.unsqueeze(0).expand(blen, -1)

            # --- PGK1 pass ---
            if has_metric:
                logits_pgk1, _, metric_pgk1 = model(lig_t, pgk1_b, return_metric_emb=True)
            else:
                logits_pgk1, _ = model(lig_t, pgk1_b)

            ligand_repr_all[i:i+blen] = _hook_store['pre_film'].numpy()   # same for both passes
            post_film_pgk1 [i:i+blen] = _hook_store['post_film'].numpy()
            p_bind_pgk1_all[i:i+blen] = torch.sigmoid(logits_pgk1).squeeze().cpu().numpy() \
                                          if blen > 1 else [torch.sigmoid(logits_pgk1).item()]

            if has_metric:
                metric_emb_all[i:i+blen] = metric_pgk1.cpu().numpy()

            # --- PGK2 pass ---
            if has_metric:
                logits_pgk2, _, _ = model(lig_t, pgk2_b, return_metric_emb=True)
            else:
                logits_pgk2, _ = model(lig_t, pgk2_b)

            post_film_pgk2 [i:i+blen] = _hook_store['post_film'].numpy()
            p_bind_pgk2_all[i:i+blen] = torch.sigmoid(logits_pgk2).squeeze().cpu().numpy() \
                                          if blen > 1 else [torch.sigmoid(logits_pgk2).item()]

            screened = min(i + batch_size, N)
            if (i // 2000) < (screened // 2000) or screened == N:
                print(f"  {screened:,} / {N:,}", flush=True)

    hook_handle.remove()
    print(f"✓ Inference complete\n", flush=True)

    # ── 7. Compute Morgan FPs ─────────────────────────────────────────────────
    print("Computing Morgan fingerprints (ECFP4, 2048-bit)...", flush=True)
    morgan_fp_all = np.zeros((N, 2048), dtype=np.uint8)
    failed_fp = 0
    for i, smi in enumerate(valid_smiles):
        fp = compute_morgan_fp(smi)
        if fp is not None:
            morgan_fp_all[i] = fp
        else:
            failed_fp += 1
    print(f"  ✓ Done  ({failed_fp} FP failures)\n", flush=True)

    selectivity_all = p_bind_pgk2_all - p_bind_pgk1_all

    # ── 8. Save outputs ───────────────────────────────────────────────────────
    print("─" * 70, flush=True)
    out_dir = Path(output_dir) if output_dir else Path(config['output']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path  = out_dir / 'projections.npz'
    meta_path = out_dir / 'projections_meta.csv'

    save_dict = dict(
        smiles         = np.array(valid_smiles),
        ligand_repr    = ligand_repr_all,        # (N, 256) pre-FiLM
        post_film_pgk1 = post_film_pgk1,         # (N, 256) FiLM(PGK1)
        post_film_pgk2 = post_film_pgk2,         # (N, 256) FiLM(PGK2)
        morgan_fp      = morgan_fp_all,           # (N, 2048)
        p_bind_pgk1    = p_bind_pgk1_all,
        p_bind_pgk2    = p_bind_pgk2_all,
        selectivity    = selectivity_all,
    )
    if has_metric:
        save_dict['metric_emb'] = metric_emb_all   # (N, 64) L2-normalized

    np.savez_compressed(str(npz_path), **save_dict)
    print(f"✓ Saved projections → {npz_path}", flush=True)

    df_meta = pd.DataFrame({
        'smiles'      : valid_smiles,
        'p_bind_pgk1' : p_bind_pgk1_all,
        'p_bind_pgk2' : p_bind_pgk2_all,
        'selectivity' : selectivity_all,
    })
    df_meta.to_csv(meta_path, index=True)   # index = row number aligns with .npz arrays
    print(f"✓ Saved metadata CSV → {meta_path}\n", flush=True)

    # Summary
    pgk2_n = (selectivity_all > 0).sum()
    pgk1_n = (selectivity_all < 0).sum()
    print("SUMMARY")
    print(f"  N compounds         : {N:,}")
    print(f"  PGK2-selective (>0) : {pgk2_n:,}  ({100*pgk2_n/N:.1f}%)")
    print(f"  PGK1-selective (<0) : {pgk1_n:,}  ({100*pgk1_n/N:.1f}%)")
    print(f"  Selectivity range   : [{selectivity_all.min():.3f}, {selectivity_all.max():.3f}]")
    print(f"\nArrays in {npz_path.name}:")
    for k, v in save_dict.items():
        if hasattr(v, 'shape'):
            print(f"  {k:20s}  shape={v.shape}  dtype={v.dtype}")
        else:
            print(f"  {k:20s}  len={len(v)}")
    print("\n✓ DONE\n", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract model projections for analysis')
    parser.add_argument('--config',       type=str, required=True,
                        help='Path to a screen_enamine YAML config')
    parser.add_argument('--output_dir',   type=str, default=None,
                        help='Override output directory from config')
    parser.add_argument('--from_results', type=str, default=None,
                        help='Path to existing screening results CSV — restrict to those SMILES')
    parser.add_argument('--top_n',        type=int, default=None,
                        help='Only process the first N rows (after loading from_results or input CSV)')
    args = parser.parse_args()

    extract(
        config_path  = args.config,
        output_dir   = args.output_dir,
        from_results = args.from_results,
        top_n        = args.top_n,
    )
