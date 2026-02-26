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
from rdkit import Chem
from rdkit.Chem import AllChem
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
# STEP 2: Generate 3D conformer (implicit H, no AddHs)
# ─────────────────────────────────────────────────────────────────────────────

def smiles_to_conformer(smiles):
    """
    Generate 3D conformer for a SMILES string WITHOUT explicit hydrogens.
    Try ETKDG first; if that fails, use random 3D coords.
    
    Returns
    -------
    mol : RdKit Mol object with 3D conformer, or None if fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Try standard ETKDG
        AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=False)
        
        # If ETKDG fails (numConfs==0), try random coords
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        
        if mol.GetNumConformers() == 0:
            return None
        
        # Try to minimize; if it fails (e.g. MMFF issues), just keep what we have
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass
        
        return mol
    except Exception as e:
        return None


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
    
    # Extract paths
    csv_paths = [Path(p) for p in config['input']['csv_paths']]
    max_smiles = config['input'].get('max_smiles', None)
    output_dir = Path(config['output']['cache_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SMILES
    print(f"Loading SMILES from {len(csv_paths)} CSV file(s)...")
    smiles_list = load_all_smiles(csv_paths, max_smiles)
    
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
    failed_smiles = []
    
    print(f"Processing {len(smiles_list)} SMILES...")
    start_time = time.time()
    
    for idx, smiles in enumerate(smiles_list):
        step_start = time.time()
        
        # Generate conformer (no explicit H)
        try:
            mol = smiles_to_conformer(smiles)
            if mol is None:
                failed_smiles.append((smiles, "conform_failed"))
                elapsed = time.time() - step_start
                if idx < 5 or (idx + 1) % 20 == 0:
                    print(f"  [{idx + 1:4d}]  FAIL conform | {elapsed:.3f}s")
                continue
        except Exception as e:
            failed_smiles.append((smiles, f"conform_error: {str(e)[:30]}"))
            elapsed = time.time() - step_start
            if idx < 5 or (idx + 1) % 20 == 0:
                print(f"  [{idx + 1:4d}]  ERROR conform | {elapsed:.3f}s | {str(e)[:40]}")
            continue
        
        # Convert to Uni-Mol input
        try:
            atoms, coords = mol_to_unimol_input(mol)
            if atoms is None or coords is None:
                failed_smiles.append((smiles, "atomization_failed"))
                elapsed = time.time() - step_start
                if idx < 5 or (idx + 1) % 20 == 0:
                    print(f"  [{idx + 1:4d}]  FAIL atoms  | {elapsed:.3f}s")
                continue
        except Exception as e:
            failed_smiles.append((smiles, f"atomization_error: {str(e)[:30]}"))
            elapsed = time.time() - step_start
            if idx < 5 or (idx + 1) % 20 == 0:
                print(f"  [{idx + 1:4d}]  ERROR atoms  | {elapsed:.3f}s | {str(e)[:40]}")
            continue
        
        # Get embedding from Uni-Mol
        try:
            repr_start = time.time()
            result = mol_repr.get_repr({'atoms': [atoms], 'coordinates': [coords]})
            repr_elapsed = time.time() - repr_start
            # result is a list of embeddings, not a dict
            cls_repr_array = np.array(result)
            embedding = cls_repr_array[0]  # extract first (only) molecule
            ligand_features[smiles] = embedding
            elapsed = time.time() - step_start
            if idx < 5 or (idx + 1) % 20 == 0:
                print(f"  [{idx + 1:4d}]  OK (repr={repr_elapsed:.3f}s, total={elapsed:.3f}s)")
        except Exception as e:
            failed_smiles.append((smiles, f"repr_error: {str(e)[:30]}"))
            elapsed = time.time() - step_start
            if idx < 5 or (idx + 1) % 20 == 0:
                print(f"  [{idx + 1:4d}]  ERROR repr   | {elapsed:.3f}s | {str(e)[:40]}")
            continue
    
    print(f"\n{'─'*70}")
    print(f"Success: {len(ligand_features):4d} / {len(smiles_list)}")
    print(f"Failed:  {len(failed_smiles):4d} / {len(smiles_list)}")
    if failed_smiles[:5]:
        print(f"  Example failed SMILES (first 5): {failed_smiles[:5]}")
    print(f"{'─'*70}\n")
    
    if not ligand_features:
        print("ERROR: No ligands successfully processed. Exiting.")
        return
    
    # Save PKL
    pkl_path = output_dir / config['output']['pkl_filename']
    with open(pkl_path, 'wb') as f:
        pickle.dump(ligand_features, f)
    print(f"Saved PKL → {pkl_path}")
    
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
