"""
Pocket-conditioned selectivity model with FiLM conditioning.

Architecture:
  1. Pocket projector: 512 → 128
  2. Ligand encoder: 512 → 256
  3. FiLM layer: pocket_proj (128) → gamma, beta for ligand (256)
  4. Classifier: 256 → 1 → sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PocketProjector(nn.Module):
    """Project frozen pocket embeddings to smaller space."""
    
    def __init__(self, pocket_dim=512, proj_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(pocket_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
    
    def forward(self, pocket_emb):
        """
        Parameters
        ----------
        pocket_emb : (batch, 512)
        
        Returns
        -------
        pocket_proj : (batch, 128)
        """
        return self.proj(pocket_emb)


class LigandEncoder(nn.Module):
    """Project Uni-Mol ligand embeddings to working dimension."""
    
    def __init__(self, ligand_dim=512, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ligand_dim, hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, ligand_feat):
        """
        Parameters
        ----------
        ligand_feat : (batch, 512)
        
        Returns
        -------
        ligand_repr : (batch, 256)
        """
        return self.proj(ligand_feat)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    
    Generates scale (gamma) and shift (beta) from pocket projection
    to modulate ligand features.
    """
    
    def __init__(self, pocket_proj_dim=128, ligand_dim=256):
        super().__init__()
        # Generate gamma and beta together
        self.film_gen = nn.Linear(pocket_proj_dim, 2 * ligand_dim)
    
    def forward(self, pocket_proj, ligand_repr):
        """
        Parameters
        ----------
        pocket_proj : (batch, 128)
        ligand_repr : (batch, 256)
        
        Returns
        -------
        modulated : (batch, 256)
        """
        film_params = self.film_gen(pocket_proj)  # (batch, 512)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # each (batch, 256)
        
        # Apply FiLM with residual bias (gamma starts near 0 → 1+gamma ≈ 1)
        modulated = (1 + gamma) * ligand_repr + beta
        return modulated


class Classifier(nn.Module):
    """Binary classifier head."""
    
    def __init__(self, input_dim=256, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch, 256)
        
        Returns
        -------
        logits : (batch, 1)
        """
        return self.head(x)


class MetricProjector(nn.Module):
    """L2-normalized projection head for metric learning.
    
    Maps ligand_repr (256-dim, pre-FiLM) to a normalized 64-dim embedding
    where PGK1-selective and PGK2-selective compounds are separated.
    """
    
    def __init__(self, input_dim=256, proj_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, proj_dim),
        )
    
    def forward(self, ligand_repr):
        """
        Parameters
        ----------
        ligand_repr : (batch, 256)  — pre-FiLM ligand encoding
        
        Returns
        -------
        emb : (batch, 64)  — L2-normalized metric embedding
        """
        emb = self.proj(ligand_repr)
        return F.normalize(emb, p=2, dim=-1)


class SelectivityModel(nn.Module):
    """
    Full pocket-conditioned selectivity prediction model.
    
    Forward pass:
      pocket_emb → pocket_proj
      ligand_feat → ligand_repr
      modulated = FiLM(pocket_proj, ligand_repr)
      logit = classifier(modulated)
      p_bind = sigmoid(logit)
    
    Optionally also returns metric embedding from MetricProjector for triplet loss.
    Branch point is controlled by metric_branch:
      'pre_film'  → MetricProjector(ligand_repr)  — pocket-blind, pure chemistry
      'post_film' → MetricProjector(modulated)    — joint (pocket, ligand) repr
    """
    
    def __init__(self,
                 pocket_dim=512,
                 pocket_proj_dim=128,
                 ligand_dim=512,
                 ligand_hidden_dim=256,
                 dropout=0.2,
                 use_metric_head=False,
                 metric_proj_dim=64,
                 metric_branch='post_film'):
        super().__init__()
        
        self.pocket_projector = PocketProjector(pocket_dim, pocket_proj_dim)
        self.ligand_encoder = LigandEncoder(ligand_dim, ligand_hidden_dim)
        self.film_layer = FiLMLayer(pocket_proj_dim, ligand_hidden_dim)
        self.classifier = Classifier(ligand_hidden_dim, hidden_dim=64, dropout=dropout)
        
        self.use_metric_head = use_metric_head
        self.metric_branch = metric_branch  # 'pre_film' or 'post_film'
        if use_metric_head:
            self.metric_projector = MetricProjector(ligand_hidden_dim, metric_proj_dim)
    
    def forward(self, ligand_feat, pocket_emb, return_metric_emb=False):
        """
        Parameters
        ----------
        ligand_feat       : (batch, 512)
        pocket_emb        : (batch, 512)
        return_metric_emb : bool — if True, also return L2-normalized metric embedding
        
        Returns
        -------
        logits     : (batch, 1)
        probs      : (batch, 1)  — sigmoid probabilities
        metric_emb : (batch, 64) — only if return_metric_emb=True and use_metric_head=True
        """
        # Project pocket
        pocket_proj = self.pocket_projector(pocket_emb)  # (batch, 128)
        
        # Encode ligand
        ligand_repr = self.ligand_encoder(ligand_feat)  # (batch, 256)
        
        # FiLM conditioning
        modulated = self.film_layer(pocket_proj, ligand_repr)  # (batch, 256)
        
        # Classify
        logits = self.classifier(modulated)  # (batch, 1)
        probs = torch.sigmoid(logits)
        
        if return_metric_emb and self.use_metric_head:
            # Branch point controlled by self.metric_branch:
            #   'pre_film'  → pocket-blind pure chemistry space
            #   'post_film' → joint (pocket, ligand) space
            src = ligand_repr if self.metric_branch == 'pre_film' else modulated
            metric_emb = self.metric_projector(src)  # (batch, 64)
            return logits, probs, metric_emb
        
        return logits, probs


if __name__ == '__main__':
    # Quick test
    print("Testing SelectivityModel...")
    
    model = SelectivityModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dummy input
    batch_size = 4
    ligand_feat = torch.randn(batch_size, 512)
    pocket_emb = torch.randn(batch_size, 512)
    
    logits, probs = model(ligand_feat, pocket_emb)
    
    print(f"\nInput shapes:")
    print(f"  ligand_feat: {ligand_feat.shape}")
    print(f"  pocket_emb:  {pocket_emb.shape}")
    print(f"\nOutput shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  probs:  {probs.shape}")
    print(f"\nSample probs: {probs.squeeze().tolist()}")
    
    print("\n✓ Model test passed")
