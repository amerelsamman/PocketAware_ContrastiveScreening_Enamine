"""
Extract pocket representations from protein-ligand PDB files using pretrained Uni-Mol.

Pipeline
--------
1. Parse each PDB file
2. Identify ligand heavy atoms (all HETATM records, excluding water HOH)
3. Extract protein residues whose Cα is within `CUTOFF` Å of ANY ligand heavy atom
4. Feed (1-letter AA codes, Cα coords) → UniMolRepr(data_type='protein')
5. Save 512-dim CLS pocket embeddings to CSV + PKL

Usage
-----
    conda activate unimol
    python extract_pocket_embeddings.py              # uses default 8.0 Å cutoff
    python extract_pocket_embeddings.py --cutoff 6   # stricter cutoff
    python extract_pocket_embeddings.py --cutoff 10  # larger shell
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PDB_DIR    = "data/full_PL_complexes"
OUTPUT_DIR = "data/pocket_embeddings_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Three-letter → one-letter amino acid map
THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    # common variants
    'HSD':'H','HSE':'H','HSP':'H','HID':'H','HIE':'H','HIP':'H',
    'MSE':'M','SEC':'C','PYL':'K',
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Parse PDB → ligand coords + protein residue Cα coords
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdb(pdb_path: str):
    """
    Returns
    -------
    ligand_coords  : np.ndarray (N_lig, 3)  — heavy atom coords of all HETATM non-water
    protein_res    : list of dicts  {resname, chain, resseq, ca_coord (3,)}
    """
    ligand_coords = []
    protein_res_map = {}   # (chain, resseq, icode) -> dict

    with open(pdb_path, 'r') as fh:
        for line in fh:
            rec = line[:6].strip()

            if rec == 'ATOM':
                chain  = line[21]
                resseq = int(line[22:26].strip())
                icode  = line[26].strip()
                resname = line[17:20].strip()
                aname   = line[12:16].strip()
                key     = (chain, resseq, icode)
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])

                if key not in protein_res_map:
                    protein_res_map[key] = {
                        'resname': resname,
                        'chain': chain,
                        'resseq': resseq,
                        'ca_coord': None,
                    }
                if aname == 'CA':
                    protein_res_map[key]['ca_coord'] = np.array([x, y, z], dtype=np.float32)

            elif rec == 'HETATM':
                resname = line[17:20].strip()
                if resname in ('HOH', 'WAT', 'H2O'):   # skip water
                    continue
                element = line[76:78].strip()
                if element == 'H':                      # skip hydrogens
                    continue
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                ligand_coords.append([x, y, z])

    ligand_coords = np.array(ligand_coords, dtype=np.float32) if ligand_coords else np.zeros((0, 3))
    protein_residues = [v for v in protein_res_map.values() if v['ca_coord'] is not None]
    return ligand_coords, protein_residues


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Filter residues within cutoff of any ligand atom
# ─────────────────────────────────────────────────────────────────────────────

def extract_pocket_residues(ligand_coords: np.ndarray,
                             protein_residues: list,
                             cutoff: float) -> list:
    """
    Returns the subset of protein_residues whose Cα is within `cutoff` Å
    of any ligand heavy atom.
    """
    if len(ligand_coords) == 0:
        print("  WARNING: no ligand heavy atoms found — returning full protein")
        return protein_residues

    pocket = []
    for res in protein_residues:
        ca = res['ca_coord']                          # (3,)
        dists = np.linalg.norm(ligand_coords - ca, axis=1)   # (N_lig,)
        if dists.min() <= cutoff:
            pocket.append(res)

    return pocket


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Convert pocket residues → UniMol input format
# ─────────────────────────────────────────────────────────────────────────────

def pocket_to_unimol_input(pocket_residues: list):
    """
    Returns
    -------
    atoms : list of 1-letter codes
    coords: np.ndarray (n, 3)
    """
    atoms  = []
    coords = []
    for res in pocket_residues:
        letter = THREE_TO_ONE.get(res['resname'].upper(), 'X')
        atoms.append(letter)
        coords.append(res['ca_coord'])
    return atoms, np.array(coords, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(cutoff: float):
    pdb_files = [f for f in os.listdir(PDB_DIR) if f.endswith('.pdb')]
    pdb_files.sort()

    print(f"\nUsing pocket cutoff: {cutoff} Å")
    print(f"Found {len(pdb_files)} PDB files in '{PDB_DIR}'\n")

    all_atoms  = []
    all_coords = []
    pocket_names = []

    for fname in pdb_files:
        name = fname.replace('.pdb', '')
        path = os.path.join(PDB_DIR, fname)

        lig_coords, prot_res = parse_pdb(path)
        pocket_res = extract_pocket_residues(lig_coords, prot_res, cutoff)
        atoms, coords = pocket_to_unimol_input(pocket_res)

        if len(atoms) == 0:
            print(f"  [{name}] WARNING: 0 pocket residues — skipping")
            continue

        print(f"  [{name}]  ligand atoms: {len(lig_coords):3d} | "
              f"pocket residues: {len(atoms):3d} | "
              f"unknown AAs: {atoms.count('X')}")

        all_atoms.append(atoms)
        all_coords.append(coords)
        pocket_names.append(name)

    if not all_atoms:
        raise RuntimeError("No pockets extracted — check PDB files and cutoff.")

    # ── Run Uni-Mol ────────────────────────────────────────────────────────
    print(f"\nLoading pretrained Uni-Mol pocket model (poc_pre_220816.pt)...")
    from unimol_tools import UniMolRepr

    repr_model = UniMolRepr(
        data_type='protein',
        remove_hs=False,
        use_cuda=False,
        model_name='unimolv1',
    )

    pocket_data = {'atoms': all_atoms, 'coordinates': all_coords}
    result      = repr_model.get_repr(pocket_data, return_atomic_reprs=True)

    cls_repr     = np.array(result['cls_repr'])   # (n_pockets, 512)
    atomic_reprs = result['atomic_reprs']         # list of (n_res, 512) arrays

    print(f"\nCLS repr shape: {cls_repr.shape}   (n_pockets × 512)")

    # ── Save PKL (full data) ───────────────────────────────────────────────
    pkl_path = os.path.join(OUTPUT_DIR, f"pocket_embeddings_unimol_cutoff{int(cutoff)}A.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'pocket_names':  pocket_names,
            'cutoff_A':      cutoff,
            'cls_repr':      cls_repr,            # (n, 512)
            'atomic_reprs':  atomic_reprs,        # list of (n_res, 512)
            'atoms':         all_atoms,
            'coords':        all_coords,
        }, f)
    print(f"Saved PKL → {pkl_path}")

    # ── Save CSV (CLS embeddings, easy to load anywhere) ──────────────────
    emb_cols = [f"emb_{i}" for i in range(cls_repr.shape[1])]
    df = pd.DataFrame(cls_repr, columns=emb_cols)
    df.insert(0, 'pocket', pocket_names)
    df.insert(1, 'label', [0 if 'PGK1' in n else 1 for n in pocket_names])
    df.insert(2, 'n_residues', [len(a) for a in all_atoms])
    df.insert(3, 'cutoff_A', cutoff)

    csv_path = os.path.join(OUTPUT_DIR, f"pocket_embeddings_unimol_cutoff{int(cutoff)}A.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV → {csv_path}")

    # ── Quick sanity print ─────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────")
    for i, name in enumerate(pocket_names):
        label = "PGK1" if "PGK1" in name else "PGK2"
        print(f"  {name:30s}  [{label}]  CLS[:6] = {cls_repr[i, :6].round(3)}")
    print("─────────────────────────────────────────────────────────────────\n")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', type=float, default=8.0,
                        help='Pocket cutoff radius in Å (default: 8.0)')
    args = parser.parse_args()
    main(args.cutoff)
