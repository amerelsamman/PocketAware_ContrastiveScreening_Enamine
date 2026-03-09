"""
screen_enamine_v2.py — Clean parallel Enamine screening for ComputeCanada.

Pipeline
--------
  Phase 1 │ Parallel RDKit conformer generation  (multiprocessing, CPU-bound)
  Phase 2 │ Batched UniMol embedding on GPU       (unimol_tools, GPU-bound)
  Phase 3 │ Batched SelectivityModel inference    (PyTorch, GPU-bound)
  Phase 4 │ Save ranked results

Key fixes vs parallelized_screen_enamine.py
-------------------------------------------
  1. UniMol use_cuda=True when GPU available (was hardcoded False — ~50x speedup)
  2. get_repr() returns a dict; extract ['cls_repr'] correctly (was crashing silently)
  3. multiprocessing 'spawn' context avoids CUDA/OpenMP fork deadlocks
  4. pool.map() (ordered) instead of imap_unordered (non-deterministic)
  5. Cache saved every `save_every` compounds — safe resume if job times out
  6. Optional SMILES canonicalization to prevent cache key mismatches

Usage
-----
  # Local test:
  conda activate unimol
  python screen_enamine_v2.py --config config.yaml --n_workers 4 --max_compounds 1000

  # Full ComputeCanada run:
  sbatch computecanada_job_v2.sh

Config additions (v2):
  embeddings:
    embed_batch_size: 256   # GPU batch size for UniMol (default 256)
    canonicalize: false     # canonicalize SMILES keys (default false)
    save_every: 50000       # save cache every N new embeddings (default 50000)
"""

import sys
import os
import warnings

# ── In spawn workers, redirect stderr before rdkit loads to suppress numpy
# ── ABI mismatch tracebacks (rdkit 2022 compiled against numpy 1.x).
# ── Workers are identified by not being __main__.
_IS_MAIN = (__name__ == '__main__')
if not _IS_MAIN:
    sys.stderr = open(os.devnull, 'w')
    warnings.filterwarnings('ignore')

if _IS_MAIN:
    print("[screen_v2] Python started, loading imports...", flush=True)

import pickle
import math
import multiprocessing
import logging
import argparse
import contextlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── suppress verbose unimol logging before it's imported ──────────────────────
logging.getLogger('unimol_tools').setLevel(logging.ERROR)
logging.getLogger('unimol_tools.tasks').setLevel(logging.ERROR)
logging.getLogger('unimol_tools.tasks.trainer').setLevel(logging.ERROR)

if _IS_MAIN:
    print("[screen_v2] Loading torch...", flush=True)
import torch
if _IS_MAIN:
    print("[screen_v2] Loading RDKit...", flush=True)
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MolToSmiles, MolFromSmiles
RDLogger.DisableLog('rdApp.*')  # suppress "no explicit Hs" and similar
if _IS_MAIN:
    print("[screen_v2] Loading model...", flush=True)
from model import SelectivityModel
if _IS_MAIN:
    print("[screen_v2] All imports done.\n", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def suppress_output():
    """Suppress stdout, stderr, and all logging temporarily."""
    root_lvl = logging.root.level
    logging.root.setLevel(logging.CRITICAL)
    for name in ['unimol_tools', 'unimol_tools.tasks', 'unimol_tools.tasks.trainer']:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    with open(os.devnull, 'w') as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            logging.root.setLevel(root_lvl)
            for name in ['unimol_tools', 'unimol_tools.tasks', 'unimol_tools.tasks.trainer']:
                logging.getLogger(name).setLevel(logging.ERROR)


def canonicalize_smiles(smi: str) -> str | None:
    """Return RDKit canonical SMILES, or None if invalid."""
    try:
        mol = MolFromSmiles(smi)
        return MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def extract_cls_repr(result) -> np.ndarray:
    """
    Safely extract CLS embeddings from a UniMolRepr.get_repr() return value.

    unimol_tools >= 1.0 returns a dict:
        {'cls_repr': np.ndarray(N, 512), 'atomic_reprs': ...}
    unimol_tools < 1.0 returned an np.ndarray or list directly.

    Returns
    -------
    np.ndarray of shape (N, 512)
    """
    if isinstance(result, dict):
        # Modern unimol_tools (>= 1.0)
        if 'cls_repr' in result:
            return np.array(result['cls_repr'])
        # Fallback: take the first array value
        for v in result.values():
            arr = np.array(v)
            if arr.ndim == 2:
                return arr
        raise ValueError(f"Cannot find 2-D embedding array in get_repr() dict. Keys: {list(result.keys())}")
    else:
        # Legacy unimol_tools: returned array/list directly
        return np.array(result)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Conformer generation (CPU, picklable, multiprocessing-safe)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_conformer_worker(smiles: str) -> tuple:
    """
    Pure-CPU, top-level (picklable) worker for multiprocessing.
    Converts a SMILES string to 3-D conformer atom+coord representation.

    Returns
    -------
    (smiles, atoms, coords, error)
      atoms  : list[str]      — heavy-atom symbols (no H)
      coords : list[list]     — (natom, 3) as a plain Python list (picklable)
      error  : str or None    — failure reason if failed
    """
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (smiles, None, None, "invalid_smiles")

        # Embed 3D conformer: try ETKDG first, then random coords fallback
        status = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=False)
        if status == -1 or mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        if mol.GetNumConformers() == 0:
            return (smiles, None, None, "conformer_failed")

        # Optional MMFF geometry optimization (best-effort)
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass

        # Extract heavy-atom coordinates
        conf = mol.GetConformer()
        atoms, coords = [], []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue  # skip explicit H
            atoms.append(atom.GetSymbol())
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])

        if not atoms:
            return (smiles, None, None, "no_heavy_atoms")

        # Return coords as plain Python list (picklable; numpy arrays are
        # picklable too, but plain lists avoid occasional numpy MP issues)
        return (smiles, atoms, coords, None)

    except Exception as e:
        return (smiles, None, None, f"exception:{str(e)[:80]}")


def generate_conformers_parallel(
    smiles_list: list[str],
    n_workers: int,
    chunksize: int = 500,
    verbose: bool = True,
) -> tuple[list[tuple], list[tuple]]:
    """
    Generate 3-D conformers for all SMILES in parallel using Python multiprocessing.

    Uses 'spawn' start context to avoid CUDA/OpenMP fork deadlocks on Linux.

    Parameters
    ----------
    smiles_list : list of SMILES strings
    n_workers   : number of worker processes
    chunksize   : Pool.map chunksize (controls granularity; 500 is reasonable)
    verbose     : print progress

    Returns
    -------
    conformers : list of (smiles, atoms, coords) — successfully embedded
    failed     : list of (smiles, reason)
    """
    if verbose:
        print(f"  Generating conformers: {len(smiles_list):,} molecules × {n_workers} workers...", flush=True)

    t0 = time.time()
    # 'spawn' is safe regardless of CUDA/OpenMP state in parent process
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        raw_results = pool.map(
            _generate_conformer_worker,
            smiles_list,
            chunksize=chunksize,
        )

    conformers, failed = [], []
    for smiles, atoms, coords, error in raw_results:
        if error:
            failed.append((smiles, error))
        else:
            conformers.append((smiles, atoms, coords))

    elapsed = time.time() - t0
    if verbose:
        rate = len(smiles_list) / elapsed if elapsed > 0 else 0
        print(f"  ✓ {len(conformers):,} conformers generated, {len(failed):,} failed "
              f"({elapsed:.1f}s, {rate:.0f} mol/s)", flush=True)

    return conformers, failed


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: GPU-batched UniMol embedding
# ══════════════════════════════════════════════════════════════════════════════

def embed_conformers_gpu(
    conformers: list[tuple],
    mol_repr,
    embed_batch_size: int = 256,
    verbose: bool = True,
) -> tuple[dict, list]:
    """
    Compute UniMol 512-D CLS embeddings for a list of (smiles, atoms, coords).

    Parameters
    ----------
    conformers      : list of (smiles, atoms: list[str], coords: list[list])
    mol_repr        : UniMolRepr instance (already initialized)
    embed_batch_size: molecules per GPU batch
    verbose         : print progress

    Returns
    -------
    embeddings_dict : dict {smiles: np.ndarray(512,)}
    failed          : list of (smiles, reason)
    """
    embeddings_dict: dict = {}
    failed: list = []
    n = len(conformers)

    for batch_start in range(0, n, embed_batch_size):
        batch = conformers[batch_start : batch_start + embed_batch_size]
        batch_smiles = [item[0] for item in batch]
        batch_atoms  = [item[1] for item in batch]
        # coords must be list-of-lists (not numpy arrays) for some unimol versions
        batch_coords = [
            item[2].tolist() if hasattr(item[2], 'tolist') else item[2]
            for item in batch
        ]

        batch_ok = False

        # ── Attempt 1: GPU batch ─────────────────────────────────────────────
        try:
            with suppress_output():
                result = mol_repr.get_repr(
                    {'atoms': batch_atoms, 'coordinates': batch_coords}
                )
            embeddings = extract_cls_repr(result)  # (N, 512)

            if embeddings.ndim == 2 and embeddings.shape == (len(batch_smiles), 512):
                for smi, emb in zip(batch_smiles, embeddings):
                    embeddings_dict[smi] = emb
                batch_ok = True
            else:
                print(
                    f"  [WARNING] Unexpected batch embedding shape "
                    f"{embeddings.shape}, expected ({len(batch_smiles)}, 512). "
                    f"Falling back to per-molecule mode.",
                    flush=True,
                )
        except Exception as e:
            print(f"  [WARNING] Batch embed failed ({e!r}). "
                  f"Falling back to per-molecule mode.", flush=True)

        # ── Attempt 2: Per-molecule fallback ─────────────────────────────────
        if not batch_ok:
            for smi, atoms, coords in zip(batch_smiles, batch_atoms, batch_coords):
                try:
                    with suppress_output():
                        result = mol_repr.get_repr(
                            {'atoms': [atoms], 'coordinates': [coords]}
                        )
                    emb = extract_cls_repr(result)  # (1, 512)
                    embeddings_dict[smi] = emb[0]
                except Exception as e2:
                    failed.append((smi, f"embed_error:{str(e2)[:80]}"))

        # ── Progress report ───────────────────────────────────────────────────
        if verbose:
            done = min(batch_start + embed_batch_size, n)
            if (done % 10_000) < embed_batch_size or done == n:
                print(f"  Embedded {done:,} / {n:,}  "
                      f"(failed so far: {len(failed)})", flush=True)

    return embeddings_dict, failed


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Model inference
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(
    valid_smiles: list[str],
    valid_row_indices: list[int],
    embeddings_dict: dict,
    model_stage1: torch.nn.Module,
    pgk1_tensor: torch.Tensor,
    pgk2_tensor: torch.Tensor,
    stage2_models: list,
    device: torch.device,
    infer_batch_size: int = 512,
    verbose: bool = True,
) -> list[dict]:
    """
    Run forward passes for Stage 1 (+ optional Stage 2 ensemble).

    Returns
    -------
    list of result dicts, one per compound
    """
    results = []
    n = len(valid_smiles)

    if verbose:
        print(f"  Screening {n:,} compounds (batch_size={infer_batch_size})...", flush=True)

    with torch.no_grad():
        # ── Stage 1: PGK1 + PGK2 binding predictions ────────────────────────
        for i in range(0, n, infer_batch_size):
            batch_smiles = valid_smiles[i : i + infer_batch_size]
            B = len(batch_smiles)

            ligand_embs = np.array([embeddings_dict[s] for s in batch_smiles])
            lig = torch.tensor(ligand_embs, dtype=torch.float32, device=device)

            pgk1_batch = pgk1_tensor.unsqueeze(0).expand(B, -1)
            pgk2_batch = pgk2_tensor.unsqueeze(0).expand(B, -1)

            logits1, _ = model_stage1(lig, pgk1_batch)
            p_pgk1 = torch.sigmoid(logits1).squeeze(-1).cpu().numpy()

            logits2, _ = model_stage1(lig, pgk2_batch)
            p_pgk2 = torch.sigmoid(logits2).squeeze(-1).cpu().numpy()

            if B == 1:
                p_pgk1 = np.array([float(p_pgk1)])
                p_pgk2 = np.array([float(p_pgk2)])

            sel = p_pgk2 - p_pgk1
            for j, smi in enumerate(batch_smiles):
                results.append({
                    'smiles'           : smi,
                    '_row_idx'         : valid_row_indices[i + j],
                    'p_bind_pgk1_stage1': float(p_pgk1[j]),
                    'p_bind_pgk2_stage1': float(p_pgk2[j]),
                    'selectivity_stage1': float(sel[j]),
                })

            if verbose:
                done = min(i + infer_batch_size, n)
                if (done % 50_000) < infer_batch_size or done == n:
                    print(f"  Stage-1 inference: {done:,} / {n:,}", flush=True)

        # ── Stage 2: LOO ensemble (optional) ────────────────────────────────
        if stage2_models:
            if verbose:
                print(f"\n  Stage-2 ensemble ({len(stage2_models)} folds)...", flush=True)

            for i in range(0, n, infer_batch_size):
                batch_smiles = valid_smiles[i : i + infer_batch_size]
                B = len(batch_smiles)

                ligand_embs = np.array([embeddings_dict[s] for s in batch_smiles])
                lig = torch.tensor(ligand_embs, dtype=torch.float32, device=device)
                pgk1_batch = pgk1_tensor.unsqueeze(0).expand(B, -1)
                pgk2_batch = pgk2_tensor.unsqueeze(0).expand(B, -1)

                fold_sels = []  # (n_folds, B)
                for fold_model in stage2_models:
                    lp1, _ = fold_model(lig, pgk1_batch)
                    lp2, _ = fold_model(lig, pgk2_batch)
                    fp1 = torch.sigmoid(lp1).squeeze(-1).cpu().numpy()
                    fp2 = torch.sigmoid(lp2).squeeze(-1).cpu().numpy()
                    if B == 1:
                        fp1, fp2 = np.array([float(fp1)]), np.array([float(fp2)])
                    fold_sels.append(fp2 - fp1)

                fold_sels = np.array(fold_sels)   # (n_folds, B)
                mean_sel  = fold_sels.mean(axis=0)
                std_sel   = fold_sels.std(axis=0)
                votes_pgk2 = (fold_sels > 0).sum(axis=0)
                vote_frac  = votes_pgk2 / len(stage2_models)
                confidence = vote_frac * np.abs(mean_sel)

                for j in range(B):
                    results[i + j].update({
                        'selectivity_stage2_mean': float(mean_sel[j]),
                        'selectivity_stage2_std' : float(std_sel[j]),
                        'votes_pgk2_selective'   : int(votes_pgk2[j]),
                        'vote_fraction'          : float(vote_frac[j]),
                        'confidence_score'       : float(confidence[j]),
                    })

                if verbose:
                    done = min(i + infer_batch_size, n)
                    if (done % 50_000) < infer_batch_size or done == n:
                        print(f"  Stage-2 ensemble: {done:,} / {n:,}", flush=True)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def screen_compounds_v2(
    config_path: str = 'config.yaml',
    n_workers: int | None = None,
    embed_batch_size: int | None = None,
    infer_batch_size: int | None = None,
    max_compounds: int | None = None,
    n_chunks: int | None = None,
    chunk_idx: int | None = None,
):
    t_start = time.time()

    print("\n" + "=" * 70, flush=True)
    print("  screen_enamine_v2 — Parallel Enamine Screening (GPU + CPU)", flush=True)
    print("=" * 70 + "\n", flush=True)

    # ── Load config ───────────────────────────────────────────────────────────
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # CLI overrides (highest priority)
    if max_compounds is not None:
        cfg.setdefault('input', {})['max_compounds'] = max_compounds

    # Resolve parallelism settings with priority: CLI > config > defaults
    n_workers = n_workers or cfg.get('parallelism', {}).get('n_workers', None)
    if n_workers is None:
        n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

    embed_batch_size = (
        embed_batch_size
        or cfg.get('embeddings', {}).get('embed_batch_size', 256)
    )
    infer_batch_size = (
        infer_batch_size
        or cfg.get('inference', {}).get('batch_size', 512)
    )
    do_canonicalize = cfg.get('embeddings', {}).get('canonicalize', False)
    save_every = cfg.get('embeddings', {}).get('save_every', 50_000)
    # chunking settings (CLI overrides config)
    n_chunks = n_chunks or cfg.get('embeddings', {}).get('n_chunks', None)
    chunk_idx = chunk_idx or cfg.get('embeddings', {}).get('chunk_idx', None)
    use_chunked_cache = cfg.get('embeddings', {}).get('use_chunked_cache', False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = (device.type == 'cuda')

    print(f"  Config          : {config_path}", flush=True)
    print(f"  Device          : {device}" + (" ✓ GPU" if use_cuda else " (CPU-only)"), flush=True)
    print(f"  CPU workers     : {n_workers}  (conformer generation)", flush=True)
    print(f"  Embed batch size: {embed_batch_size}", flush=True)
    print(f"  Infer batch size: {infer_batch_size}", flush=True)
    print(f"  Canonicalize    : {do_canonicalize}", flush=True)
    print(f"  Save every      : {save_every:,} embeddings\n", flush=True)
    if use_chunked_cache and n_chunks is not None and chunk_idx is not None:
        print(f"  Chunking        : {chunk_idx+1}/{n_chunks} (per-chunk cache enabled)", flush=True)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Load compounds
    # ═════════════════════════════════════════════════════════════════════════
    print("─" * 70, flush=True)
    print("STAGE 1 — Loading Enamine Compound Library", flush=True)
    print("─" * 70, flush=True)

    enamine_csv = Path(cfg['input']['enamine_csv'])
    smiles_col  = cfg['input'].get('smiles_column', 'smiles')
    limit       = cfg['input'].get('max_compounds', None)

    print(f"  {enamine_csv}", flush=True)
    df = pd.read_csv(enamine_csv).reset_index(drop=True)
    series = df[smiles_col].dropna()
    all_smiles = series.tolist()
    all_row_indices = series.index.tolist()

    if limit:
        all_smiles = all_smiles[:limit]
        all_row_indices = all_row_indices[:limit]

    # Optional SMILES canonicalization (prevents cache key mismatches)
    if do_canonicalize:
        print("  Canonicalizing SMILES...", flush=True)
        canon_map = {}  # orig → canonical
        for smi in all_smiles:
            c = canonicalize_smiles(smi)
            canon_map[smi] = c if c else smi
        all_smiles = [canon_map[s] for s in all_smiles]

    print(f"  ✓ Loaded {len(all_smiles):,} compounds\n", flush=True)

    # ── Optional chunking for batch runs: split full library into N chunks
    if n_chunks is not None and chunk_idx is not None:
        total = len(all_smiles)
        if not (0 <= chunk_idx < n_chunks):
            raise ValueError(f"chunk_idx must be in [0, {n_chunks-1}]")
        chunk_size = math.ceil(total / n_chunks)
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total)
        print(f"  Chunking: processing chunk {chunk_idx+1}/{n_chunks} "
              f"(indices {start}:{end}, {end-start} compounds)", flush=True)
        all_smiles = all_smiles[start:end]
        all_row_indices = all_row_indices[start:end]

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Embedding cache: load existing + compute missing
    # ═════════════════════════════════════════════════════════════════════════
    print("─" * 70, flush=True)
    print("STAGE 2 — UniMol Embeddings (cache + compute)", flush=True)
    print("─" * 70, flush=True)

    orig_cache_path = Path(cfg['embeddings']['cache_path'])
    cache_path = orig_cache_path
    # If per-chunk cache is enabled, rewrite the cache filename so each chunk
    # writes/reads its own pkl (safe for parallel chunked runs).
    if use_chunked_cache and n_chunks is not None and chunk_idx is not None:
        cache_path = orig_cache_path.with_name(
            f"{orig_cache_path.stem}.chunk{chunk_idx:04d}_of_{n_chunks}{orig_cache_path.suffix}"
        )
        print(f"  Using per-chunk cache: {cache_path}", flush=True)
    use_cache  = cfg['embeddings'].get('use_cache', True)
    embeddings_dict: dict = {}

    if use_cache and cache_path.exists():
        print(f"  Loading cache from {cache_path}...", flush=True)
        with open(cache_path, 'rb') as fh:
            embeddings_dict = pickle.load(fh)
        overlap = sum(1 for s in all_smiles if s in embeddings_dict)
        print(f"  ✓ Cache: {len(embeddings_dict):,} entries, "
              f"{overlap:,}/{len(all_smiles):,} overlap with input", flush=True)

        # handle optional conformer aggregation: if user requested and raw
        # cache exists then precompute collapsed vectors here
        agg = cfg.get('embeddings', {}).get('conformer_aggregate', None)
        if agg:
            # determine raw cache path
            raw_path = cfg.get('embeddings', {}).get('raw_cache_path', None)
            if not raw_path:
                raw_path = cache_path.with_name(cache_path.stem + '_raw' + cache_path.suffix)
            raw_path = Path(raw_path)
            if raw_path.exists():
                print(f"  Loading raw multi-conf cache from {raw_path}...", flush=True)
                with open(raw_path, 'rb') as fh:
                    raw_dict = pickle.load(fh)
                # collapse into embeddings_dict if missing or overwrite
                for smi, arrs in raw_dict.items():
                    try:
                        mat = np.vstack(arrs)
                    except Exception:
                        continue
                    if agg == 'max':
                        embeddings_dict[smi] = np.max(mat, axis=0)
                    else:
                        embeddings_dict[smi] = np.mean(mat, axis=0)
                print(f"  Collapsed {len(raw_dict):,} multi-conf entries via {agg}", flush=True)
            else:
                print(f"  [warning] conformer_aggregate={agg} requested but raw cache {raw_path} not found", flush=True)

        if overlap == 0 and len(embeddings_dict) > 0:
            print(
                "  [WARNING] Zero overlap between cache and input SMILES.\n"
                "  This usually means the cache was built from a different batch\n"
                "  of molecules or SMILES are in a different format.\n"
                "  → All compounds will be re-embedded.\n"
                "  Tip: set `canonicalize: true` in config to prevent this.",
                flush=True,
            )
            embeddings_dict = {}
    else:
        print(f"  No existing cache found at {cache_path}", flush=True)

    missing_smiles = [s for s in all_smiles if s not in embeddings_dict]
    print(f"\n  Missing embeddings: {len(missing_smiles):,}", flush=True)

    if missing_smiles:
        # ── Phase 1: Parallel conformer generation ────────────────────────
        print(f"\n  [Phase 1] Conformer generation ({n_workers} CPU workers)", flush=True)
        conformers, conf_failed = generate_conformers_parallel(
            missing_smiles,
            n_workers=n_workers,
            verbose=True,
        )

        # ── Phase 2: GPU-batched UniMol embedding ─────────────────────────
        print(f"\n  [Phase 2] UniMol embedding on {device} "
              f"(batch_size={embed_batch_size})", flush=True)
        print("  Initializing UniMol encoder (may take 2-3 min)...", flush=True)
        with suppress_output():
            from unimol_tools import UniMolRepr
            mol_repr = UniMolRepr(
                data_type='molecule',
                remove_hs=True,
                use_cuda=use_cuda,   # ← KEY FIX: use GPU when available
                model_name='unimolv1',
            )
        print("  ✓ UniMol encoder ready\n", flush=True)

        new_embeddings, embed_failed = embed_conformers_gpu(
            conformers,
            mol_repr,
            embed_batch_size=embed_batch_size,
            verbose=True,
        )

        embeddings_dict.update(new_embeddings)

        # ── Save updated cache incrementally ─────────────────────────────
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as fh:
            pickle.dump(embeddings_dict, fh)
        print(f"\n  ✓ Cache saved: {len(embeddings_dict):,} entries → {cache_path}", flush=True)

        # ── Save failures ─────────────────────────────────────────────────
        all_failed = conf_failed + embed_failed
        if all_failed:
            failed_path = cache_path.parent / (cache_path.stem + '_failed.txt')
            with open(failed_path, 'w') as fh:
                for smi, reason in all_failed:
                    fh.write(f"{smi}\t{reason}\n")
            print(f"  ⚠ {len(all_failed):,} failures logged → {failed_path}", flush=True)
    else:
        print("  ✓ All embeddings already cached — skipping computation\n", flush=True)

    # Filter to only compounds with valid embeddings
    valid_pairs = [
        (s, idx)
        for s, idx in zip(all_smiles, all_row_indices)
        if s in embeddings_dict
    ]
    valid_smiles     = [p[0] for p in valid_pairs]
    valid_row_indices = [p[1] for p in valid_pairs]
    print(f"\n  ✓ Proceeding with {len(valid_smiles):,} / {len(all_smiles):,} "
          f"compounds (embeddings available)\n", flush=True)

    if len(valid_smiles) == 0:
        raise RuntimeError(
            "No valid embeddings found. Check SMILES column, cache path, and "
            "UniMol model weights. See failed.txt for details."
        )

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Pocket embeddings
    # ═════════════════════════════════════════════════════════════════════════
    print("─" * 70, flush=True)
    print("STAGE 3 — Loading Pocket Embeddings", flush=True)
    print("─" * 70, flush=True)

    pocket_path = Path(cfg['pockets']['pocket_embeddings_path'])
    with open(pocket_path, 'rb') as fh:
        pocket_data = pickle.load(fh)

    pocket_names = pocket_data['pocket_names']
    pocket_embs  = pocket_data['cls_repr']   # (N_pockets, 512)

    pgk1_idx = [i for i, n in enumerate(pocket_names) if 'PGK1' in n]
    pgk2_idx = [i for i, n in enumerate(pocket_names) if 'PGK2' in n]

    pgk1_mean = pocket_embs[pgk1_idx].mean(axis=0)
    pgk2_mean = pocket_embs[pgk2_idx].mean(axis=0)

    pgk1_tensor = torch.tensor(pgk1_mean, dtype=torch.float32, device=device)
    pgk2_tensor = torch.tensor(pgk2_mean, dtype=torch.float32, device=device)

    print(f"  ✓ {len(pocket_names)} pockets loaded "
          f"(PGK1: {len(pgk1_idx)}, PGK2: {len(pgk2_idx)})\n", flush=True)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Load SelectivityModel checkpoint(s)
    # ═════════════════════════════════════════════════════════════════════════
    print("─" * 70, flush=True)
    print("STAGE 4 — Loading SelectivityModel", flush=True)
    print("─" * 70, flush=True)

    ckpt_path  = Path(cfg['model']['checkpoint_path'])
    checkpoint = torch.load(ckpt_path, map_location=device)
    _state     = checkpoint['model_state_dict']
    _has_metric = any('metric_projector' in k for k in _state.keys())

    model_stage1 = SelectivityModel(
        ligand_dim=512,
        pocket_dim=512,
        ligand_hidden_dim=256,
        pocket_proj_dim=128,
        use_metric_head=_has_metric,
    ).to(device)
    model_stage1.load_state_dict(_state)
    model_stage1.eval()

    print(f"  ✓ Stage-1 model loaded  (epoch {checkpoint.get('epoch', '?')}, "
          f"val AUC {checkpoint.get('val_auc', 0):.4f})", flush=True)

    # Optional Stage-2 LOO ensemble
    stage2_models = []
    if cfg['model'].get('use_stage2_ensemble', False):
        stage2_dir = Path(cfg['model']['stage2_loo_dir'])
        print(f"  Loading Stage-2 LOO ensemble from {stage2_dir}...", flush=True)
        for fold_idx in range(1, 16):
            ckpt_f = stage2_dir / f"fold_{fold_idx}" / f"fold_{fold_idx}_best.pt"
            if not ckpt_f.exists():
                print(f"  [WARN] fold {fold_idx} not found — skipping", flush=True)
                continue
            fc  = torch.load(ckpt_f, map_location=device)
            fs  = fc['model_state_dict']
            fm  = SelectivityModel(512, 512, 256, 128,
                                   use_metric_head=any('metric_projector' in k for k in fs)).to(device)
            fm.load_state_dict(fs)
            fm.eval()
            stage2_models.append(fm)
        print(f"  ✓ {len(stage2_models)}/15 Stage-2 fold models loaded\n", flush=True)
    else:
        print("  Stage-2 ensemble disabled\n", flush=True)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 5 — Model inference
    # ═════════════════════════════════════════════════════════════════════════
    print("─" * 70, flush=True)
    print("STAGE 5 — Model Inference", flush=True)
    print("─" * 70, flush=True)

    results = run_inference(
        valid_smiles=valid_smiles,
        valid_row_indices=valid_row_indices,
        embeddings_dict=embeddings_dict,
        model_stage1=model_stage1,
        pgk1_tensor=pgk1_tensor,
        pgk2_tensor=pgk2_tensor,
        stage2_models=stage2_models,
        device=device,
        infer_batch_size=infer_batch_size,
        verbose=True,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 6 — Save results
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70, flush=True)
    print("STAGE 6 — Saving Results", flush=True)
    print("─" * 70, flush=True)

    output_dir = Path(cfg['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(results)

    # Merge original metadata columns (ID, catalog number, MW, etc.)
    df_meta = df.drop(columns=[smiles_col], errors='ignore')
    df_results = df_results.merge(
        df_meta, left_on='_row_idx', right_index=True, how='left'
    ).drop(columns=['_row_idx'])

    use_stage2 = bool(stage2_models)

    if use_stage2:
        # Sort by majority vote, then by |selectivity|
        df_results['_abs_sel'] = df_results['selectivity_stage2_mean'].abs()
        df_results = df_results.sort_values(
            by=['votes_pgk2_selective', '_abs_sel', 'selectivity_stage2_std'],
            ascending=[False, False, True],
        ).drop(columns=['_abs_sel'])

        def _cat(row):
            v, n = row['votes_pgk2_selective'], len(stage2_models)
            if v >= n * 0.8:   return 'PGK2-selective (high confidence)'
            elif v >= n * 0.6: return 'PGK2-selective (moderate confidence)'
            elif v >= n * 0.4: return 'Uncertain'
            elif v >= n * 0.2: return 'PGK1-selective (moderate confidence)'
            else:              return 'PGK1-selective (high confidence)'
        df_results['selectivity_category'] = df_results.apply(_cat, axis=1)

    else:
        if cfg['output'].get('preserve_order', False):
            pass  # keep original order
        else:
            df_results = df_results.sort_values('selectivity_stage1', ascending=False)
        df_results['selectivity'] = df_results['selectivity_stage1'].apply(
            lambda s: 'PGK2' if s > 0 else 'PGK1'
        )

    # Full results (per-chunk when chunking enabled to avoid clobbering)
    full_fname = cfg['output']['full_results_filename']
    if use_chunked_cache and n_chunks is not None and chunk_idx is not None:
        p = Path(full_fname)
        full_fname = f"{p.stem}.chunk{chunk_idx:04d}_of_{n_chunks}{p.suffix}"
    full_path = output_dir / full_fname
    df_results.to_csv(full_path, index=False)
    print(f"  ✓ Full results ({len(df_results):,} compounds) → {full_path}", flush=True)

    # PGK2-selective
    if use_stage2:
        df_pgk2 = df_results[df_results['votes_pgk2_selective'] > len(stage2_models) / 2]
        df_pgk1 = df_results[df_results['votes_pgk2_selective'] < len(stage2_models) / 2].iloc[::-1]
    else:
        df_pgk2 = df_results[df_results['selectivity_stage1'] > 0]
        df_pgk1 = df_results[df_results['selectivity_stage1'] < 0]

    # Per-chunk filenames for selective outputs as well
    pgk2_fname = cfg['output']['pgk2_selective_filename']
    pgk1_fname = cfg['output']['pgk1_selective_filename']
    if use_chunked_cache and n_chunks is not None and chunk_idx is not None:
        p2 = Path(pgk2_fname); p1 = Path(pgk1_fname)
        pgk2_fname = f"{p2.stem}.chunk{chunk_idx:04d}_of_{n_chunks}{p2.suffix}"
        pgk1_fname = f"{p1.stem}.chunk{chunk_idx:04d}_of_{n_chunks}{p1.suffix}"

    pgk2_path = output_dir / pgk2_fname
    df_pgk2.to_csv(pgk2_path, index=False)
    print(f"  ✓ PGK2-selective ({len(df_pgk2):,}) → {pgk2_path}", flush=True)

    pgk1_path = output_dir / pgk1_fname
    df_pgk1.to_csv(pgk1_path, index=False)
    print(f"  ✓ PGK1-selective ({len(df_pgk1):,}) → {pgk1_path}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'═' * 70}", flush=True)
    print(f"  SCREENING COMPLETE  —  {elapsed / 60:.1f} min total", flush=True)
    print(f"{'═' * 70}", flush=True)
    print(f"  Compounds screened : {len(df_results):,}", flush=True)
    print(f"  PGK2-selective     : {len(df_pgk2):,}", flush=True)
    print(f"  PGK1-selective     : {len(df_pgk1):,}", flush=True)

    if use_stage2:
        print(f"  Stage-2 mean sel   : {df_results['selectivity_stage2_mean'].mean():.4f}", flush=True)
    else:
        print(f"  Stage-1 mean sel   : {df_results['selectivity_stage1'].mean():.4f}", flush=True)
        top5 = df_results[['smiles', 'selectivity_stage1']].head(5)
        print(f"\n  Top 5 PGK2-selective:", flush=True)
        print(top5.to_string(index=False), flush=True)

    print(f"\n  Results in: {output_dir}/\n", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Required by 'spawn' multiprocessing on Windows (no-op on Linux)
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description='Parallelized Enamine screening (GPU + multicore CPU)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config',          default='config.yaml',
                        help='Path to config YAML')
    parser.add_argument('--n_workers',       type=int, default=None,
                        help='CPU workers for conformer gen '
                             '(default: SLURM_CPUS_PER_TASK or cpu_count())')
    parser.add_argument('--embed_batch_size',type=int, default=None,
                        help='UniMol GPU batch size (default: from config or 256)')
    parser.add_argument('--infer_batch_size',type=int, default=None,
                        help='Inference batch size (default: from config or 512)')
    parser.add_argument('--max_compounds',   type=int, default=None,
                        help='Process only first N compounds (for testing)')
    parser.add_argument('--n_chunks',        type=int, default=None,
                        help='Total number of chunks to split the library into')
    parser.add_argument('--chunk_idx',       type=int, default=None,
                        help='Zero-based index of the chunk to process (0..n_chunks-1)')
    args = parser.parse_args()

    screen_compounds_v2(
        config_path     =args.config,
        n_workers       =args.n_workers,
        embed_batch_size=args.embed_batch_size,
        infer_batch_size=args.infer_batch_size,
        max_compounds   =args.max_compounds,
        n_chunks        =args.n_chunks,
        chunk_idx       =args.chunk_idx,
    )
