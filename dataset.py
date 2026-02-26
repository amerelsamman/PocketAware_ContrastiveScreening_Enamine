"""
Dataset for pocket-conditioned binding/selectivity prediction.

Stages:
  Stage 1 (binding): Learn to distinguish binders from non-binders
    - Use both PGK1_mean and PGK2_mean pockets for all ligands
  Stage 2 (selectivity): Learn isoform specificity on PDB ligands with known targets
    - Use target_pocket (positive) and opposite_pocket (negative)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# LIGAND ↔ POCKET MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# Map ligand_id to the PDB structure it comes from
LIGAND_TO_STRUCTURE = {
    'L1': ('PGK2_cmp21', 'PGK2'),      # cmp21 = PGK2 selective
    'L2': ('PGK2_cmp47', 'PGK2'),      # cmp47 = PGK2 selective
    'L3': ('PGK1_cmp45', 'PGK1'),      # cmp45 = PGK1 selective
    # Add more as known
}


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SelectivityDataset(Dataset):
    """
    Dataset for pocket-conditioned binding & selectivity prediction.
    
    Parameters
    ----------
    stage : str
        'binding' = use all ligands with both pockets (binding classification)
        'selectivity' = use only PDB ligands with known targets (selectivity)
        'combined' = concatenate both stages
    """
    
    def __init__(self, stage: str = 'binding', csv_path: str = None):
        """
        Load pockets, ligand features, and ligand metadata.
        """
        self.stage = stage
        assert stage in ('binding', 'selectivity', 'combined'), \
            f"stage must be 'binding', 'selectivity', or 'combined', got {stage}"
        
        # Load pocket embeddings
        with open('data/pocket_embeddings_cache/pocket_embeddings_unimol_cutoff8A.pkl', 'rb') as f:
            pocket_data = pickle.load(f)
        
        self.pocket_names = pocket_data['pocket_names']  # ['PGK1_4O33', 'PGK1_4O3F', ...]
        self.pocket_embeddings = pocket_data['cls_repr']  # (6, 512)
        
        # Compute pocket means
        pgk1_indices = [i for i, name in enumerate(self.pocket_names) if 'PGK1' in name]
        pgk2_indices = [i for i, name in enumerate(self.pocket_names) if 'PGK2' in name]
        
        self.pgk1_mean = self.pocket_embeddings[pgk1_indices].mean(axis=0)  # (512,)
        self.pgk2_mean = self.pocket_embeddings[pgk2_indices].mean(axis=0)  # (512,)
        
        # Create pocket index mapping
        self.pocket_name_to_idx = {name: i for i, name in enumerate(self.pocket_names)}
        
        print(f"Loaded {len(self.pocket_names)} pockets")
        print(f"  PGK1_mean: {self.pgk1_mean.shape}, PGK2_mean: {self.pgk2_mean.shape}")
        
        # Load ligand features
        with open('data/ligand_features_cache/ligand_features_unimol.pkl', 'rb') as f:
            self.ligand_features = pickle.load(f)  # {smiles: (512,)}
        
        print(f"Loaded {len(self.ligand_features)} ligand embeddings")
        
        # Load ligand metadata
        if csv_path is not None:
            csv_paths = [Path(csv_path)]
        else:
            csv_paths = [Path('data/ligands/smiles_binding.csv')]
        
        dfs = []
        for path in csv_paths:
            if path.exists():
                df = pd.read_csv(path)
                dfs.append(df)
        
        self.metadata = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        print(f"Loaded {len(self.metadata)} ligand metadata rows")
        
        # Create training examples
        self.examples = []
        
        if stage in ('binding', 'combined'):
            self._create_binding_examples()
        
        if stage in ('selectivity', 'combined'):
            self._create_selectivity_examples()
        
        print(f"Created {len(self.examples)} training examples for stage={stage}")
    
    def _create_binding_examples(self):
        """
        Stage 1: Create (ligand_feat, pocket_emb, label) for binding vs. non-binding.
        
        DEL hits (selectivity=1): PGK2-selective binders
          → (ligand, PGK2_mean, label=1) ← binds PGK2
          → (ligand, PGK1_mean, label=0) ← doesn't bind PGK1 (selective)
        
        Decoys (selectivity=0): non-binders to both
          → (ligand, PGK1_mean, label=0)
          → (ligand, PGK2_mean, label=0)
        
        PDB ligands: depends on known target
          → if target=PGK2: (ligand, PGK2_mean, label=1), (ligand, PGK1_mean, label=0)
          → if target=PGK1: (ligand, PGK1_mean, label=1), (ligand, PGK2_mean, label=0)
        """
        print("\nCreating binding examples (Stage 1)...")
        
        for _, row in self.metadata.iterrows():
            smiles = row['smiles']
            selectivity = row.get('selectivity', None)
            source = row.get('source', 'unknown')
            target = row.get('target', None)
            
            # Skip if no SMILES or feature
            if pd.isna(smiles) or smiles not in self.ligand_features:
                continue
            
            ligand_feat = self.ligand_features[smiles]
            
            # ─── DEL hits (selectivity=1): PGK2-selective ───
            if source == 'DEL' and selectivity == 1:
                # Binds PGK2
                self.examples.append({
                    'smiles': smiles,
                    'ligand_feat': ligand_feat.copy(),
                    'pocket_emb': self.pgk2_mean.copy(),
                    'label': 1,
                    'stage': 'binding',
                    'source': 'DEL',
                })
                # Doesn't bind PGK1 (selective)
                self.examples.append({
                    'smiles': smiles,
                    'ligand_feat': ligand_feat.copy(),
                    'pocket_emb': self.pgk1_mean.copy(),
                    'label': 0,
                    'stage': 'binding',
                    'source': 'DEL',
                })
            
            # ─── Decoys (selectivity=0): non-binders ───
            elif (source == 'DECOY') or (source == 'DEL' and selectivity == 0):
                # Don't bind either
                self.examples.append({
                    'smiles': smiles,
                    'ligand_feat': ligand_feat.copy(),
                    'pocket_emb': self.pgk1_mean.copy(),
                    'label': 0,
                    'stage': 'binding',
                    'source': 'DECOY',
                })
                self.examples.append({
                    'smiles': smiles,
                    'ligand_feat': ligand_feat.copy(),
                    'pocket_emb': self.pgk2_mean.copy(),
                    'label': 0,
                    'stage': 'binding',
                    'source': 'DECOY',
                })
            
            # ─── PDB ligands (with known target) ───
            elif source == 'PDB' and pd.notna(target):
                if target == 'PGK2':
                    # Binds PGK2
                    self.examples.append({
                        'smiles': smiles,
                        'ligand_feat': ligand_feat.copy(),
                        'pocket_emb': self.pgk2_mean.copy(),
                        'label': 1,
                        'stage': 'binding',
                        'source': 'PDB',
                        'target': 'PGK2',
                    })
                    # Doesn't bind PGK1
                    self.examples.append({
                        'smiles': smiles,
                        'ligand_feat': ligand_feat.copy(),
                        'pocket_emb': self.pgk1_mean.copy(),
                        'label': 0,
                        'stage': 'binding',
                        'source': 'PDB',
                        'target': 'PGK2',
                    })
                elif target == 'PGK1':
                    # Binds PGK1
                    self.examples.append({
                        'smiles': smiles,
                        'ligand_feat': ligand_feat.copy(),
                        'pocket_emb': self.pgk1_mean.copy(),
                        'label': 1,
                        'stage': 'binding',
                        'source': 'PDB',
                        'target': 'PGK1',
                    })
                    # Doesn't bind PGK2
                    self.examples.append({
                        'smiles': smiles,
                        'ligand_feat': ligand_feat.copy(),
                        'pocket_emb': self.pgk2_mean.copy(),
                        'label': 0,
                        'stage': 'binding',
                        'source': 'PDB',
                        'target': 'PGK1',
                    })
        
        print(f"  → {len([e for e in self.examples if e['stage']=='binding'])} binding examples")
    
    def _create_selectivity_examples(self):
        """
        Stage 2: Refine selectivity on PDB ligands with known targets.
        
        For each PDB ligand with known target:
          → if target=PGK2: (ligand, PGK2_mean, label=1), (ligand, PGK1_mean, label=0)
          → if target=PGK1: (ligand, PGK1_mean, label=1), (ligand, PGK2_mean, label=0)
        """
        print("\nCreating selectivity examples (Stage 2)...")
        
        for _, row in self.metadata.iterrows():
            smiles = row['smiles']
            source = row.get('source', None)
            target = row.get('target', None)
            
            # Only use PDB ligands with known targets
            if source != 'PDB' or pd.isna(target):
                continue
            
            # Skip if no SMILES or feature
            if pd.isna(smiles) or smiles not in self.ligand_features:
                continue
            
            ligand_feat = self.ligand_features[smiles]
            
            # Assign pockets based on target
            if target == 'PGK1':
                # Binds PGK1
                self.examples.append({
                    'smiles': smiles,
                    'ligand_feat': ligand_feat.copy(),
                    'pocket_emb': self.pgk1_mean.copy(),
                    'label': 1,
                    'stage': 'selectivity',
                    'target': target,
                    'source': 'PDB',
                })
                # Doesn't bind PGK2
                self.examples.append({
                    'smiles': smiles,
                    'ligand_feat': ligand_feat.copy(),
                    'pocket_emb': self.pgk2_mean.copy(),
                    'label': 0,
                    'stage': 'selectivity',
                    'target': target,
                    'source': 'PDB',
                })
            elif target == 'PGK2':
                # Binds PGK2
                self.examples.append({
                    'smiles': smiles,
                    'ligand_feat': ligand_feat.copy(),
                    'pocket_emb': self.pgk2_mean.copy(),
                    'label': 1,
                    'stage': 'selectivity',
                    'target': target,
                    'source': 'PDB',
                })
                # Doesn't bind PGK1
                self.examples.append({
                    'smiles': smiles,
                    'ligand_feat': ligand_feat.copy(),
                    'pocket_emb': self.pgk1_mean.copy(),
                    'label': 0,
                    'stage': 'selectivity',
                    'target': target,
                    'source': 'PDB',
                })
        
        print(f"  → {len([e for e in self.examples if e['stage']=='selectivity'])} selectivity examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx) -> Dict:
        """
        Returns
        -------
        dict with keys:
          'ligand_feat' : (512,)
          'pocket_emb'  : (512,)
          'label'       : 0 or 1
          'source'      : 'PDB', 'DEL', or 'DECOY'
                    'smiles'      : SMILES string
        """
        ex = self.examples[idx]
        return {
            'ligand_feat': torch.tensor(ex['ligand_feat'], dtype=torch.float32),
            'pocket_emb': torch.tensor(ex['pocket_emb'], dtype=torch.float32),
            'label': torch.tensor(ex['label'], dtype=torch.long),
            'source': ex.get('source', 'unknown'),
            'target': ex.get('target', ''),  # empty string for non-PDB examples
                        'smiles': ex.get('smiles', ''),
        }


if __name__ == '__main__':
    # Quick test
    print("Testing binding stage...")
    ds_binding = SelectivityDataset(stage='binding')
    print(f"Dataset length: {len(ds_binding)}\n")
    
    ex = ds_binding[0]
    print(f"Example sample shapes:")
    print(f"  ligand_feat: {ex['ligand_feat'].shape}")
    print(f"  pocket_emb:  {ex['pocket_emb'].shape}")
    print(f"  label:       {ex['label']}")
    
    print("\nTesting selectivity stage...")
    ds_selectivity = SelectivityDataset(stage='selectivity')
    print(f"Dataset length: {len(ds_selectivity)}\n")
    
    print("\nTesting combined stage...")
    ds_combined = SelectivityDataset(stage='combined')
    print(f"Dataset length: {len(ds_combined)}\n")
