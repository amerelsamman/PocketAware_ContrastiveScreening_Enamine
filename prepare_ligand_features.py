"""
Prepare ligand features using pretrained Uni-Mol molecule encoder.

Pipeline
--------
1. Load all SMILES from CSVs (deduplicate)
2. For each SMILES:
   - Parse with RDKit  (implicit H, no AddHs)
   - Generate 3D conformer (ETKDG)
   - Run UniMolRepr(data_type='molecule', remove_hs=True) → 512-dim CLS
3. Save to PKL + CSV

Usage
-----
    conda activate unimol
    python prepare_ligand_features.py --config config_prepare_features.yaml
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import time
import argparse
import yaml
import logging
from rdkit import Chem
from rdkit.Chem import AllChem

# suppress unimol_tools logging noise during feature generation
logging.getLogger('unimol_tools').setLevel(logging.CRITICAL)
logging.getLogger('unimol_tools.tasks').setLevel(logging.CRITICAL)
logging.getLogger('unimol_tools.tasks.trainer').setLevel(logging.CRITICAL)

from unimol_tools import UniMolRepr

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load & deduplicate SMILES
# ─────────────────────────────────────────────────────────────────────────────

def load_all_smiles(csv_paths, max_smiles=None):
    """
    Load SMILES from multiple CSVs, deduplicate.
    
    Returns
    -------
    smiles_list : list of str
    """
    all_smiles = set()
    
    for path in csv_paths:
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        
        df = pd.read_csv(path)
        smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
        
        valid = df[smiles_col].dropna().unique()
        all_smiles.update(valid)
        print(f"  [{path.name}] loaded {len(valid)} rows, {len(all_smiles)} cumulative unique SMILES")
    
    smiles_list = sorted(list(all_smiles))
    if max_smiles:
        smiles_list = smiles_list[:max_smiles]
    
    print(f"\nTotal unique SMILES: {len(smiles_list)}\n")
    return smiles_list


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Generate 3D conformers (implicit H, no AddHs)
# ─────────────────────────────────────────────────────────────────────────────


def smiles_to_conformers(smiles, n_confs=1):
    """
    Generate up to ``n_confs`` 3D conformers for a SMILES string.
    Uses ETKDG embedding with fallback to random coords; if multiple
    conformers are requested we call ``EmbedMultipleConfs``.  Returns a
    list of RDKit Mol objects each carrying exactly one conformer.

    Parameters
    ----------
    smiles : str
    n_confs : int
        Number of conformers to attempt (>=1).

    Returns
    -------
    list[Chem.Mol]
        Possibly empty if embedding failed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    # ``EmbedMultipleConfs`` retains all conformers in a single Mol
    # object; we will split them afterwards into separate Mol copies.
    try:
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=42)
    except Exception:
        ids = []
    if not ids:
        # fallback to single-conformer embed (ETKDG then random)
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=False)
        except Exception:
            pass
        if mol.GetNumConformers() == 0:
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            except Exception:
                pass

    # minimize each conformer if possible
    try:
        for cid in range(mol.GetNumConformers()):
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=cid)
            except Exception:
                pass
    except Exception:
        pass

    # split into separate molecules
    mols = []
    for cid in range(mol.GetNumConformers()):
        m2 = Chem.Mol(mol)
        # keep only this conformer
        for other in list(range(m2.GetNumConformers())):
            if other != cid:
                m2.RemoveConformer(other)
        mols.append(m2)
    return mols


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Extract atoms & coords for Uni-Mol
# ─────────────────────────────────────────────────────────────────────────────

def mol_to_unimol_input(mol):
    """
    Convert RdKit Mol with conformer to Uni-Mol input (atoms + coords).
    
    Returns
    -------
    atoms  : list of element symbols (no H)
    coords : np.ndarray (n_atoms, 3)
    """
    if mol.GetNumConformers() == 0:
        return None, None
    
    conf = mol.GetConformer()
    atoms = []
    coords = []
    
    for atom in mol.GetAtoms():
        # Skip H
        if atom.GetAtomicNum() == 1:
            continue
        atoms.append(atom.GetSymbol())
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    
    if not atoms:
        return None, None
    
    return atoms, np.array(coords, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path='config_prepare_features.yaml'):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*70}")
    print("Prepare Ligand Features via Uni-Mol Molecule Encoder")
    print(f"Config: {config_path}")
    print(f"{'='*70}\n")
    
    # Extract paths and configuration parameters
    csv_paths = [Path(p) for p in config['input']['csv_paths']]
    max_smiles = config['input'].get('max_smiles', None)
    output_dir = Path(config['output']['cache_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    conf_cfg = config.get('conformer', {})
    n_confs = conf_cfg.get('n_confs', 1)
    collapse_method = conf_cfg.get('collapse', 'mean')
    pdb_dir = Path(conf_cfg.get('pdb_dir', ''))
    save_multiconf = config['output'].get('save_multiconf', False)

    # Load SMILES
    print(f"Loading SMILES from {len(csv_paths)} CSV file(s)...")
    smiles_list = load_all_smiles(csv_paths, max_smiles)
    
    # Build mapping from SMILES -> pdb filename (description column)
    pdb_lookup = {}
    csv_meta = Path(csv_paths[0]) if csv_paths else None
    if csv_meta and csv_meta.exists():
        try:
            df_meta = pd.read_csv(csv_meta)
            if 'description' in df_meta.columns and 'smiles' in df_meta.columns:
                for _, row in df_meta.iterrows():
                    smi = row.get('smiles')
                    desc = row.get('description')
                    if pd.notna(smi) and pd.notna(desc):
                        pdb_lookup[smi] = desc
        except Exception:
            pass

    # Initialize Uni-Mol
    print(f"Loading pretrained Uni-Mol molecule encoder...")
    unimol_params = config['unimol']
    mol_repr = UniMolRepr(
        data_type=unimol_params['data_type'],
        remove_hs=unimol_params['remove_hs'],
        use_cuda=False,
        model_name='unimolv1',
    )
    print()

    # Process each SMILES
    ligand_features = {}  # smiles -> embedding (512,)
    multiconf_features = {}  # smiles -> list of embeddings if save_multiconf
    failed_smiles = []

    print(f"Processing {len(smiles_list)} SMILES...")
    start_time = time.time()
    
    for idx, smiles in enumerate(smiles_list):
        step_start = time.time()

        # determine conformers: check for PDB geometry first
        conf_mols = []
        pdb_file = pdb_lookup.get(smiles)
        if pdb_file and pdb_dir:
            # some descriptions omit the .pdb extension – tolerate both forms
            pdb_path = pdb_dir / pdb_file
            if not pdb_path.suffix.lower() == '.pdb':
                alt = pdb_path.with_suffix('.pdb')
                if alt.exists():
                    pdb_path = alt
            if pdb_path.exists():
                m = Chem.MolFromPDBFile(str(pdb_path), removeHs=True)
                if m is not None and m.GetNumConformers() > 0:
                    # keep the crystallographic pose as one conformer
                    conf_mols = [m]
                    # also generate the remaining ETKDG poses so that every
                    # molecule ends up with ``n_confs`` total embeddings.
                    if n_confs > 1:
                        extra = smiles_to_conformers(smiles, n_confs=n_confs - 1)
                        conf_mols.extend(extra)
        if not conf_mols:
            # standard generation (PDB not found or failed)
            conf_mols = smiles_to_conformers(smiles, n_confs=n_confs)

        if not conf_mols:
            failed_smiles.append((smiles, "conform_failed"))
            elapsed = time.time() - step_start
            if idx < 5 or (idx + 1) % 1000 == 0:
                print(f"  [{idx + 1:,}]  FAIL conform | {elapsed:.3f}s")
            continue

        # convert each conformer and get embedding
        embeddings = []
        for mol in conf_mols:
            try:
                atoms, coords = mol_to_unimol_input(mol)
                if atoms is None or coords is None:
                    raise ValueError("atomization_failed")
            except Exception as e:
                failed_smiles.append((smiles, f"atomization_failed"))
                embeddings = []
                break

            try:
                repr_start = time.time()
                # temporarily disable all logging to silence UniMol messages
                prev = logging.root.manager.disable
                logging.disable(logging.CRITICAL)
                result = mol_repr.get_repr({'atoms': [atoms], 'coordinates': [coords]})
                logging.disable(prev)
                repr_elapsed = time.time() - repr_start
                cls_repr_array = np.array(result)
                embeddings.append(cls_repr_array[0])
            except Exception as e:
                failed_smiles.append((smiles, f"repr_error"))
                embeddings = []
                break

        if not embeddings:
            elapsed = time.time() - step_start
            if idx < 5 or (idx + 1) % 1000 == 0:
                print(f"  [{idx + 1:4d}]  FAIL embed | {elapsed:.3f}s")
            continue

        # collapse embeddings
        if collapse_method == 'max':
            emb = np.max(np.vstack(embeddings), axis=0)
        else:
            emb = np.mean(np.vstack(embeddings), axis=0)
        ligand_features[smiles] = emb
        if save_multiconf:
            multiconf_features[smiles] = embeddings

        elapsed = time.time() - step_start
        if idx < 5 or (idx + 1) % 1000 == 0:
            print(f"  [{idx + 1:,}]  OK (confs={len(embeddings)}, total={elapsed:.3f}s)")
    
    print(f"\n{'─'*70}")
    print(f"Success: {len(ligand_features):4d} / {len(smiles_list)}")
    print(f"Failed:  {len(failed_smiles):4d} / {len(smiles_list)}")
    if failed_smiles[:5]:
        print(f"  Example failed SMILES (first 5): {failed_smiles[:5]}")
    print(f"{'─'*70}\n")
    
    if not ligand_features:
        print("ERROR: No ligands successfully processed. Exiting.")
        return
    
    # Save PKL (collapsed features)
    pkl_path = output_dir / config['output']['pkl_filename']
    with open(pkl_path, 'wb') as f:
        pickle.dump(ligand_features, f)
    print(f"Saved PKL → {pkl_path}")

    # Optionally save per-conformer embeddings
    if save_multiconf and multiconf_features:
        multi_path = output_dir / (config['output']['pkl_filename'].rsplit('.',1)[0] + '_confs.pkl')
        with open(multi_path, 'wb') as f:
            pickle.dump(multiconf_features, f)
        print(f"Saved multi-conformation embeddings → {multi_path}")
    
    # Save CSV
    smiles_list_success = sorted(ligand_features.keys())
    embeddings_array = np.array([ligand_features[s] for s in smiles_list_success])
    
    emb_cols = [f"emb_{i}" for i in range(512)]
    df = pd.DataFrame(embeddings_array, columns=emb_cols)
    df.insert(0, 'smiles', smiles_list_success)
    
    csv_path = output_dir / config['output']['csv_filename']
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV → {csv_path}\n")
    
    # Save failed SMILES for reference
    if failed_smiles:
        failed_path = output_dir / "ligand_features_failed.txt"
        with open(failed_path, 'w') as f:
            for smi in failed_smiles:
                f.write(smi + '\n')
        print(f"Saved failed SMILES → {failed_path}\n")
    
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare ligand features using Uni-Mol')
    parser.add_argument('--config', type=str, default='config_prepare_features.yaml',
                       help='Path to config YAML file')
    args = parser.parse_args()
    main(config_path=args.config)
