"""
Standalone ligand evaluation script for a given data source.
Loads an existing checkpoint and outputs predictions for ligands from a
specified source (PDB by default, but e.g. DECOY).

The original `eval_pdb.py` was hardcoded to the PDB subset; this version
accepts `--source` and will write results named accordingly.

Usage:
    python eval_pdb.py --config config_stage1_v1.yaml --source DECOY
"""
import argparse
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import SelectivityDataset
from model import SelectivityModel


def evaluate_source_predictions(model, dataset, checkpoint_path, results_dir, device, source='PDB'):
    """Generate binding/selectivity predictions for ligands of ``source``.

    The original function assumed ``source='PDB'`` and produced exactly 30
    rows (15 ligands × 2 pockets).  With another source (e.g. ``DECOY``)
    the number of rows may differ.  ``source`` is used for filtering,
    printing and output filenames.
    """
    print("\n" + "=" * 60)
    print(f"{source} LIGAND PREDICTIONS")
    print("=" * 60)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    pgk1_mean = torch.tensor(dataset.pgk1_mean, dtype=torch.float32).unsqueeze(0).to(device)
    pgk2_mean = torch.tensor(dataset.pgk2_mean, dtype=torch.float32).unsqueeze(0).to(device)

    meta = dataset.metadata[dataset.metadata['source'] == source].copy()

    print(f"  {len(meta)} rows selected from metadata")

    long_rows    = []
    summary_rows = []

    for _, row in meta.iterrows():
        smi = row['smiles']
        if pd.isna(smi) or smi not in dataset.ligand_features:
            print(f"  ⚠ skipping {row.get('ligand_id','')} — no feature available")
            continue

        feat = torch.tensor(dataset.ligand_features[smi], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            _, p1 = model(feat, pgk1_mean)
            _, p2 = model(feat, pgk2_mean)

        p_pgk1 = float(p1.item())
        p_pgk2 = float(p2.item())
        true_target = row.get('target', '')

        base = {
            'mol_id'     : row.get('mol_id', ''),
            'ligand_id'  : row.get('ligand_id', ''),
            'description': row.get('description', ''),
            'smiles'     : smi,
            'true_target': true_target,
        }

        # Row 1: PGK1 pocket
        tl_pgk1 = 1 if true_target == 'PGK1' else 0
        long_rows.append({**base,
            'pocket'    : 'PGK1',
            'true_label': tl_pgk1,
            'p_bind'    : round(p_pgk1, 4),
            'pred_label': int(p_pgk1 > 0.5),
            'correct'   : int(p_pgk1 > 0.5) == tl_pgk1,
        })

        # Row 2: PGK2 pocket
        tl_pgk2 = 1 if true_target == 'PGK2' else 0
        long_rows.append({**base,
            'pocket'    : 'PGK2',
            'true_label': tl_pgk2,
            'p_bind'    : round(p_pgk2, 4),
            'pred_label': int(p_pgk2 > 0.5),
            'correct'   : int(p_pgk2 > 0.5) == tl_pgk2,
        })

        # Summary row
        score = p_pgk2 - p_pgk1
        pred_target = 'PGK2' if score > 0 else 'PGK1'
        summary_rows.append({**base,
            'true_selectivity'      : row.get('selectivity', ''),
            'p_bind_pgk1'           : round(p_pgk1, 4),
            'p_bind_pgk2'           : round(p_pgk2, 4),
            'pred_selectivity_score': round(score, 4),
            'pred_target'           : pred_target,
            'correct_selectivity'   : pred_target == true_target,
        })

    df_long    = pd.DataFrame(long_rows).sort_values(['true_target', 'mol_id', 'pocket']).reset_index(drop=True)
    df_summary = pd.DataFrame(summary_rows).sort_values(['true_target', 'mol_id']).reset_index(drop=True)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    out_long    = results_dir / f'{source.lower()}_ligand_predictions.csv'
    out_summary = results_dir / f'{source.lower()}_ligand_predictions_summary.csv'
    df_long.to_csv(out_long, index=False)
    df_summary.to_csv(out_summary, index=False)

    # ── Console print ─────────────────────────────────────────────────────────
    print(df_long[['ligand_id', 'description', 'true_target', 'pocket',
                   'true_label', 'p_bind', 'pred_label', 'correct']].to_string(index=False))

    print("\n  ── Binding prediction accuracy (per pocket) ──")
    for pocket in ['PGK1', 'PGK2']:
        sub = df_long[df_long['pocket'] == pocket]
        print(f"    {pocket} pocket: {sub['correct'].sum()}/{len(sub)} correct  ({100*sub['correct'].mean():.1f}%)")
    total_rows = len(df_long)
    print(f"    Overall ({total_rows} rows): {df_long['correct'].sum()}/{total_rows} correct  "
          f"({100*df_long['correct'].mean():.1f}%)")

    n_ligands = len(df_summary)
    print(f"\n  ── Selectivity classification accuracy ({n_ligands} ligands) ──")
    for tgt in ['PGK1', 'PGK2']:
        sub = df_summary[df_summary['true_target'] == tgt]
        acc = sub['correct_selectivity'].mean() if len(sub) else float('nan')
        print(f"    {tgt}: {sub['correct_selectivity'].sum()}/{len(sub)} correct  ({100*acc:.1f}%)")
    overall = df_summary['correct_selectivity'].mean()
    print(f"    Overall: {df_summary['correct_selectivity'].sum()}/{len(df_summary)} correct  ({100*overall:.1f}%)")

    print(f"\n  Saved → {out_long}")
    print(f"  Saved → {out_summary}")

    return df_long, df_summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate ligand predictions from a saved checkpoint')
    parser.add_argument('--config', type=str, default='config_stage1_v1.yaml')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path (default: read from config)')
    parser.add_argument('--source', type=str, default='PDB',
                        help='metadata source to evaluate (e.g. PDB, DECOY)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    version      = config.get('version', 'v1')
    results_dir  = Path(config['output']['results_dir'])
    ckpt_name    = config['output']['checkpoint_name']
    ckpt_path    = args.checkpoint or str(Path(config['output']['checkpoint_dir']) / version / ckpt_name)

    device_cfg = config['training']['device']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_cfg == 'auto' else torch.device(device_cfg)
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    metric_cfg     = config.get('metric_learning', {})
    model = SelectivityModel(
        use_metric_head=metric_cfg.get('enabled', False),
        metric_proj_dim=metric_cfg.get('proj_dim', 64),
        metric_branch=metric_cfg.get('branch', 'post_film'),
    ).to(device)

    dataset = SelectivityDataset(stage=config['training']['stage'],
                                 csv_path=config['input']['ligand_csv'])

    evaluate_source_predictions(model, dataset, ckpt_path, results_dir, device,
                                source=args.source)


if __name__ == '__main__':
    main()
