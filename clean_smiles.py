#!/usr/bin/env python3
"""Clean SMILES to remove stereochemistry and unsupported atoms for REINVENT4."""

from rdkit import Chem
from pathlib import Path

input_path = Path("reinvent4_configs/pgk2_seeds.smi")
output_path = Path("reinvent4_configs/pgk2_seeds_clean.smi")

# REINVENT prior supported atoms: C, N, O, S, F, Cl, Br (NO I, P, etc.)
SUPPORTED_ATOMS = {'C', 'N', 'O', 'S', 'F', 'Cl', 'Br'}

def has_unsupported_atoms(mol):
    """Check if molecule has atoms not supported by REINVENT prior."""
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in SUPPORTED_ATOMS:
            return True, symbol
    return False, None

smiles_list = []
skipped = []

with open(input_path) as f:
    for line in f:
        smi = line.strip()
        if not smi:
            continue
        
        # Canonicalize and remove stereochemistry
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            skipped.append((smi, "invalid SMILES"))
            continue
        
        # Check for unsupported atoms
        has_unsupported, atom = has_unsupported_atoms(mol)
        if has_unsupported:
            skipped.append((smi, f"unsupported atom: {atom}"))
            continue
        
        # Remove stereochemistry
        Chem.RemoveStereochemistry(mol)
        clean_smi = Chem.MolToSmiles(mol)
        smiles_list.append(clean_smi)

# Write cleaned SMILES
with open(output_path, 'w') as f:
    for smi in smiles_list:
        f.write(smi + '\n')

print(f"✓ Kept {len(smiles_list)} SMILES (removed stereochemistry)")
print(f"✗ Skipped {len(skipped)} SMILES")
if skipped:
    print("\nSkipped examples:")
    for smi, reason in skipped[:5]:
        print(f"  - {smi}: {reason}")
print(f"\nSaved to {output_path}")
print(f"First 5 cleaned: {smiles_list[:5]}")
