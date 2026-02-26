"""
MaxMin diversity picking from a screening results set.

Strategy
--------
1. Load projections.npz (Morgan FPs + selectivity scores).
2. Medicinal chemistry pre-filter:
   - Remove PAINS (pan-assay interference, all A/B/C sets)
   - Remove reactive / undesirable groups (acyl halides, Michael acceptors, etc.)
   - Cap sulfur atom count (default <= 2)
   - Optional Lipinski / Veber drug-likeness filters
3. Filter to a high-selectivity pool (top POOL_TOP_PCT or POOL_TOP_N).
4. Score-weighted MaxMin diversity picking:
     composite(i) = alpha * norm_selectivity(i) + (1-alpha) * min_tanimoto_dist(i, selected)
   - alpha=0.0 → pure MaxMin (structural diversity only)
   - alpha=1.0 → pure greedy top-score (no diversity)
   - alpha=0.5 → balanced (default)
5. Export selected compounds to CSV with full annotations.

Usage
-----
    python pick_diverse.py \\
        --npz  data/enamine/screening_v1/projections.npz \\
        --meta data/enamine/screening_v1/projections_meta.csv \\
        --clusters data/enamine/screening_v1/butina_clusters_0.6_all.csv \\
        --n_pick 150 \\
        --pool_top_pct 30 \\
        --score_weight 0.5 \\
        --max_sulfur 2 \\
        --pains_filter \\
        --out data/enamine/screening_v1/top150_diverse.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


# ─────────────────────────────────────────────────────────────────────────────
# Chemistry filters
# ─────────────────────────────────────────────────────────────────────────────

def build_pains_catalog():
    """Build RDKit PAINS filter catalog (all 3 PAINS sets A/B/C)."""
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


# SMARTS for common reactive / undesirable groups
REACTIVE_SMARTS = [
    ('[F,Cl,Br,I][CX4]',           'alkyl halide'),
    ('C(=O)[F,Cl,Br,I]',           'acyl halide'),
    ('[N;R0][N;R0]',                'hydrazine'),
    ('O=C-O-C=O',                   'anhydride'),
    ('C=C-C=O',                     'Michael acceptor (enone)'),
    ('[SH]',                        'free thiol'),
    ('N=[N+]=[N-]',                 'azide'),
    ('[N+](=O)[O-]',                'nitro'),
    ('C1(=O)OC(=O)1',              'beta-lactone'),
    ('[#6]-O-O-[#6]',              'peroxide'),
]
REACTIVE_MOLS = [(Chem.MolFromSmarts(s), name) for s, name in REACTIVE_SMARTS]


def passes_chemistry_filters(mol, pains_catalog,
                              max_sulfur=2,
                              apply_pains=True,
                              apply_reactive=True,
                              apply_lipinski=False):
    """
    Returns (passes: bool, reason: str).
    reason is '' if passes, or a short description of the first failure.
    """
    if mol is None:
        return False, 'invalid SMILES'

    # ── Sulfur count ──────────────────────────────────────────────────────────
    n_sulfur = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
    if n_sulfur > max_sulfur:
        return False, f'sulfur_count={n_sulfur}'

    # ── PAINS ─────────────────────────────────────────────────────────────────
    if apply_pains and pains_catalog.HasMatch(mol):
        entry = pains_catalog.GetFirstMatch(mol)
        return False, f'PAINS:{entry.GetDescription()}'

    # ── Reactive groups ───────────────────────────────────────────────────────
    if apply_reactive:
        for pat, name in REACTIVE_MOLS:
            if pat is not None and mol.HasSubstructMatch(pat):
                return False, f'reactive:{name}'

    # ── Lipinski / Veber drug-likeness (optional) ────────────────────────────
    if apply_lipinski:
        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = Descriptors.TPSA(mol)
        rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
        if mw   > 500: return False, f'MW={mw:.0f}'
        if logp > 5:   return False, f'logP={logp:.1f}'
        if hbd  > 5:   return False, f'HBD={hbd}'
        if hba  > 10:  return False, f'HBA={hba}'
        if tpsa > 140: return False, f'TPSA={tpsa:.0f}'
        if rotb > 10:  return False, f'RotBonds={rotb}'

    return True, ''


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint helper
# ─────────────────────────────────────────────────────────────────────────────

def morgan_fp_to_rdkit(arr):
    """Convert uint8 numpy bit-array → RDKit ExplicitBitVect."""
    fp = DataStructs.ExplicitBitVect(len(arr))
    for b in np.where(arr)[0].tolist():
        fp.SetBit(b)
    return fp


# ─────────────────────────────────────────────────────────────────────────────
# Score-weighted MaxMin picking
# ─────────────────────────────────────────────────────────────────────────────

def score_weighted_maxmin(rdkit_fps, norm_scores, seed_idx, n_pick, alpha=0.5, verbose=True):
    """
    Score-weighted MaxMin diversity picking.

    At each step, selects the compound i* that maximises:
        alpha * norm_score(i) + (1 - alpha) * min_tanimoto_dist(i, selected)

    Parameters
    ----------
    rdkit_fps   : list of ExplicitBitVect — pool fingerprints
    norm_scores : np.ndarray (N,) — selectivity scores normalised to [0, 1]
    seed_idx    : int — index of the seed compound (highest selectivity)
    n_pick      : int — total picks including seed
    alpha       : float — weight for score vs diversity (0=pure diversity, 1=pure score)

    Returns
    -------
    selected : list of int — indices into rdkit_fps / norm_scores
    """
    N = len(rdkit_fps)
    n_pick = min(n_pick, N)

    # min_sim[i] = min Tanimoto similarity to any selected compound so far
    min_sim = np.ones(N, dtype=np.float32)
    selected = []
    remaining = set(range(N))

    # ── Seed ─────────────────────────────────────────────────────────────────
    selected.append(seed_idx)
    remaining.discard(seed_idx)

    sims = np.array(DataStructs.BulkTanimotoSimilarity(rdkit_fps[seed_idx], rdkit_fps),
                    dtype=np.float32)
    min_sim = np.minimum(min_sim, sims)

    if verbose:
        print(f"  Seed: idx={seed_idx}  norm_score={norm_scores[seed_idx]:.3f}")

    # ── Greedy iterations ────────────────────────────────────────────────────
    for step in range(1, n_pick):
        rem_arr   = np.array(list(remaining), dtype=np.int32)
        min_dist  = 1.0 - min_sim[rem_arr]          # Tanimoto distance to set
        composite = alpha * norm_scores[rem_arr] + (1.0 - alpha) * min_dist

        best_local = int(np.argmax(composite))
        best_idx   = int(rem_arr[best_local])

        selected.append(best_idx)
        remaining.discard(best_idx)

        # Update min_sim
        sims = np.array(DataStructs.BulkTanimotoSimilarity(rdkit_fps[best_idx], rdkit_fps),
                        dtype=np.float32)
        min_sim = np.minimum(min_sim, sims)

        if verbose and (step % 25 == 0 or step < 5):
            dist  = float(1.0 - min_sim[best_idx])
            score = float(norm_scores[best_idx])
            print(f"  Pick {step+1:3d} | idx={best_idx:6d} | "
                  f"norm_sel={score:.3f}  min_dist={dist:.3f}  composite={composite[best_local]:.3f}")

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Score-weighted MaxMin diversity picking with chemistry filters')
    # Data  (use --npz OR --csv, not both)
    parser.add_argument('--npz',     default=None,
                        help='Path to projections.npz (morgan_fp + selectivity keys)')
    parser.add_argument('--csv',     default=None,
                        help='Path to screening results CSV (SMILES + selectivity columns)')
    parser.add_argument('--sel_col', default='selectivity_stage1',
                        help='Column name for selectivity score in --csv (default: selectivity_stage1)')
    parser.add_argument('--meta',         default='data/enamine/screening_v1/projections_meta.csv')
    parser.add_argument('--clusters',     default=None,
                        help='Butina cluster CSV for annotation (optional)')
    # Picking
    parser.add_argument('--n_pick',       type=int,   default=150)
    parser.add_argument('--pool_top_pct', type=float, default=30.0,
                        help='Pre-filter to top N%% by selectivity (default: 30)')
    parser.add_argument('--pool_top_n',   type=int,   default=None,
                        help='Override: absolute top-N pool (ignores pool_top_pct)')
    parser.add_argument('--score_weight', type=float, default=0.5,
                        help='alpha: 0=pure diversity, 1=pure score (default: 0.5)')
    # Chemistry filters
    parser.add_argument('--max_sulfur',      type=int,  default=2,
                        help='Max sulfur atoms allowed (default: 2)')
    parser.add_argument('--pains_filter',    action='store_true', default=True,
                        help='Remove PAINS compounds (default: on)')
    parser.add_argument('--no_pains',        action='store_true',
                        help='Disable PAINS filter')
    parser.add_argument('--reactive_filter', action='store_true', default=True,
                        help='Remove reactive groups (default: on)')
    parser.add_argument('--no_reactive',     action='store_true',
                        help='Disable reactive group filter')
    parser.add_argument('--lipinski',        action='store_true', default=False,
                        help='Apply Lipinski/Veber drug-likeness filters (default: off)')
    # Output
    parser.add_argument('--out', default='data/enamine/screening_v1/top150_diverse.csv')
    args = parser.parse_args()

    apply_pains    = args.pains_filter and not args.no_pains
    apply_reactive = args.reactive_filter and not args.no_reactive

    # validate input
    if args.csv is None and args.npz is None:
        args.npz = 'data/enamine/screening_v1/projections.npz'   # legacy default
    if args.csv and args.npz:
        parser.error('Provide --npz OR --csv, not both')

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    df_input   = None   # extra columns to carry through (CSV mode only)
    extra_cols = []     # names of extra columns to include in output

    if args.csv:
        # ── CSV mode: load screening results, compute Morgan FPs on the fly ──
        df_input = pd.read_csv(args.csv)
        if args.sel_col not in df_input.columns:
            raise ValueError(f"Column '{args.sel_col}' not found in {args.csv}. "
                             f"Available: {list(df_input.columns)}")
        smiles_all  = df_input['smiles'].tolist()
        selectivity = df_input[args.sel_col].values.astype(np.float32)

        # Identify extra non-SMILES, non-selectivity columns to pass through
        skip = {'smiles', args.sel_col}
        extra_cols = [c for c in df_input.columns if c not in skip]

        print(f"  Computing Morgan FPs for {len(smiles_all):,} compounds...")
        gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        fp_list = []
        for smi in smiles_all:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                arr = np.zeros(2048, dtype=np.uint8)
                bv  = gen.GetFingerprint(mol)
                DataStructs.ConvertToNumpyArray(bv, arr)
                fp_list.append(arr)
            else:
                fp_list.append(np.zeros(2048, dtype=np.uint8))
        morgan_fp = np.vstack(fp_list)
        # npz-style stub so downstream code stays unchanged
        data = {'p_bind_pgk1': None, 'p_bind_pgk2': None}   # may be overridden below
        for col, key in [('p_bind_pgk1_stage1', 'p_bind_pgk1'),
                         ('p_bind_pgk2_stage1', 'p_bind_pgk2')]:
            if col in df_input.columns:
                data[key] = df_input[col].values.astype(np.float32)
        data['files'] = set(k for k, v in data.items() if v is not None)
    else:
        # ── NPZ mode ─────────────────────────────────────────────────────────
        raw  = np.load(args.npz, allow_pickle=True)
        smiles_all  = raw['smiles'].tolist()
        morgan_fp   = raw['morgan_fp']
        selectivity = raw['selectivity']
        data = raw

    N = len(smiles_all)
    print(f"  {N:,} compounds, selectivity [{selectivity.min():.3f}, {selectivity.max():.3f}]")

    # ── Step 1: selectivity-based pre-pool (broad, before chemistry filters) ──
    # We take a larger pre-pool (3× target) so that after chemistry filtering
    # we still have enough candidates to pick n_pick from.
    if args.pool_top_n is not None:
        prepool_size = min(args.pool_top_n * 3, N)
    else:
        prepool_size = max(args.n_pick * 3,
                           int(np.ceil(N * args.pool_top_pct / 100.0)))
    prepool_size = min(prepool_size, N)

    prepool_indices = np.argsort(selectivity)[::-1][:prepool_size]
    print(f"\nPre-pool: top {prepool_size:,} compounds by selectivity "
          f"(range [{selectivity[prepool_indices].min():.3f}, "
          f"{selectivity[prepool_indices].max():.3f}])")

    # ── Step 2: chemistry pre-filter (applied only to pre-pool) ──────────────
    print(f"\nApplying chemistry filters to pre-pool "
          f"(max_sulfur={args.max_sulfur}, PAINS={apply_pains}, "
          f"reactive={apply_reactive}, Lipinski={args.lipinski})...")

    pains_cat    = build_pains_catalog()
    passed_list  = []
    fail_reasons = {}
    mols_cache   = {}

    for gi in prepool_indices:
        smi = smiles_all[gi]
        mol = Chem.MolFromSmiles(smi)
        mols_cache[gi] = mol
        ok, reason = passes_chemistry_filters(
            mol, pains_cat,
            max_sulfur=args.max_sulfur,
            apply_pains=apply_pains,
            apply_reactive=apply_reactive,
            apply_lipinski=args.lipinski,
        )
        if ok:
            passed_list.append(gi)
        else:
            cat = reason.split('=')[0].split(':')[0]
            fail_reasons[cat] = fail_reasons.get(cat, 0) + 1

    n_pass = len(passed_list)
    n_fail = prepool_size - n_pass
    print(f"  Passed: {n_pass:,}  Failed: {n_fail:,}")
    if fail_reasons:
        for reason, cnt in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {cnt:,}")

    passed_indices = np.array(passed_list, dtype=np.int64)

    # ── Step 3: final pool — top N% of chemistry-passing compounds ────────────
    passed_sel = selectivity[passed_indices]
    if args.pool_top_n is not None:
        pool_size = min(args.pool_top_n, len(passed_indices))
    else:
        pool_size = max(args.n_pick, int(np.ceil(len(passed_indices) * args.pool_top_pct / 100.0)))

    top_local    = np.argsort(passed_sel)[::-1][:pool_size]
    pool_indices = passed_indices[top_local]   # global indices
    pool_sel     = selectivity[pool_indices]

    print(f"\nPool: top {args.pool_top_pct:.0f}% of filtered = {pool_size:,} compounds")
    print(f"  Selectivity range in pool: [{pool_sel.min():.3f}, {pool_sel.max():.3f}]")

    # ── Normalise selectivity scores to [0, 1] for composite criterion ────────
    s_min, s_max = pool_sel.min(), pool_sel.max()
    norm_scores  = (pool_sel - s_min) / (s_max - s_min + 1e-9)

    # ── Convert fingerprints ──────────────────────────────────────────────────
    print("\nConverting fingerprints...")
    pool_fps = [morgan_fp_to_rdkit(morgan_fp[i]) for i in pool_indices]

    # ── Score-weighted MaxMin ─────────────────────────────────────────────────
    print(f"\nRunning score-weighted MaxMin "
          f"(n_pick={args.n_pick}, alpha={args.score_weight})...")
    selected_local = score_weighted_maxmin(
        pool_fps, norm_scores,
        seed_idx=0,        # seed = highest selectivity (pool is sorted descending)
        n_pick=args.n_pick,
        alpha=args.score_weight,
        verbose=True,
    )
    selected_global = [pool_indices[i] for i in selected_local]

    # ── Compute per-compound descriptors ─────────────────────────────────────
    def mol_props(mol):
        if mol is None:
            return {'MW': None, 'logP': None, 'HBD': None, 'HBA': None,
                    'TPSA': None, 'RotBonds': None, 'n_sulfur': None, 'n_rings': None}
        return {
            'MW'      : round(Descriptors.MolWt(mol), 1),
            'logP'    : round(Descriptors.MolLogP(mol), 2),
            'HBD'     : rdMolDescriptors.CalcNumHBD(mol),
            'HBA'     : rdMolDescriptors.CalcNumHBA(mol),
            'TPSA'    : round(Descriptors.TPSA(mol), 1),
            'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'n_sulfur': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16),
            'n_rings' : rdMolDescriptors.CalcNumRings(mol),
        }

    # ── Build output DataFrame ────────────────────────────────────────────────
    def get_bind(key, idx):
        arr = data[key] if isinstance(data, dict) else (data[key] if key in data.files else None)
        if arr is None:
            return None
        return round(float(arr[idx]), 4)

    rows = []
    for rank, (local_i, global_i) in enumerate(zip(selected_local, selected_global), start=1):
        smi   = smiles_all[global_i]
        props = mol_props(mols_cache.get(global_i))
        row   = {
            'pick_rank'        : rank,
            'smiles'           : smi,
            'selectivity_score': round(float(selectivity[global_i]), 4),
            'p_bind_pgk1'      : get_bind('p_bind_pgk1', global_i),
            'p_bind_pgk2'      : get_bind('p_bind_pgk2', global_i),
            **props,
        }
        # carry through extra CSV columns (mol_id, catalogue ID, etc.)
        if df_input is not None:
            for col in extra_cols:
                row[col] = df_input.iloc[global_i][col]
        rows.append(row)

    df_out = pd.DataFrame(rows)

    # Annotate with cluster info
    if args.clusters and Path(args.clusters).exists():
        cl = pd.read_csv(args.clusters)[['smiles', 'cluster_id', 'cluster_size', 'cluster_mean_selectivity']]
        df_out = df_out.merge(cl, on='smiles', how='left')
        print(f"\n  Annotated with cluster IDs from {args.clusters}")

    # ── Intra-set Tanimoto stats ──────────────────────────────────────────────
    print("\nComputing intra-selection Tanimoto statistics...")
    sel_fps  = [pool_fps[i] for i in selected_local]
    tan_vals = []
    for i in range(len(sel_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(sel_fps[i], sel_fps)
        tan_vals.extend([s for j, s in enumerate(sims) if j != i])
    tan_arr = np.array(tan_vals)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DIVERSITY PICKING COMPLETE")
    print(f"{'='*60}")
    print(f"  Selected                 : {len(df_out)}")
    print(f"  Selectivity range        : [{df_out['selectivity_score'].min():.3f}, {df_out['selectivity_score'].max():.3f}]")
    print(f"  Mean selectivity         : {df_out['selectivity_score'].mean():.3f}")
    print(f"  Intra-set Tanimoto       : mean={tan_arr.mean():.3f}  max={tan_arr.max():.3f}  median={np.median(tan_arr):.3f}")
    if 'n_sulfur' in df_out.columns:
        print(f"  Mean sulfur atoms        : {df_out['n_sulfur'].mean():.2f}  (max={df_out['n_sulfur'].max()})")
    if 'cluster_id' in df_out.columns:
        print(f"  Unique clusters covered  : {df_out['cluster_id'].nunique()}")
    print(f"\n  Saved → {out_path}")

    display_cols = [c for c in ['pick_rank', 'smiles', 'selectivity_score',
                                'p_bind_pgk1', 'p_bind_pgk2',
                                'MW', 'logP', 'n_sulfur',
                                'cluster_id', 'cluster_size']
                    if c in df_out.columns]
    print(f"\nTop 20 picks:")
    print(df_out[display_cols].head(20).to_string(index=False))


if __name__ == '__main__':
    main()
