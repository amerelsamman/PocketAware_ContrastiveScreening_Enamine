"""
Filter a screening results CSV to only PGK2-selective compounds.

Usage
-----
    python filter_pgk2.py path/to/results.csv
    python filter_pgk2.py path/to/results.csv --out path/to/output.csv
    python filter_pgk2.py path/to/results.csv --col selectivity_stage1   # different column name
    python filter_pgk2.py data/enamine/screening_v1/kinaselib_screening_0-100000_results.csv
"""

import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description='Filter screening CSV to PGK2-selective rows only')
parser.add_argument('input', type=str, help='Path to input CSV')
parser.add_argument('--out', type=str, default=None,
                    help='Output path (default: <input_stem>_pgk2.csv next to input)')
parser.add_argument('--col', type=str, default='selectivity',
                    help='Column name to filter on (default: selectivity)')
args = parser.parse_args()

inp = Path(args.input)
out = Path(args.out) if args.out else inp.parent / (inp.stem + '_pgk2.csv')

df = pd.read_csv(inp)
print(f"Loaded {len(df):,} rows from {inp}")

if args.col not in df.columns:
    raise SystemExit(f"Column '{args.col}' not found. Available columns: {list(df.columns)}")

before = len(df)
df_pgk2 = df[df[args.col] == 'PGK2']
after = len(df_pgk2)

df_pgk2.to_csv(out, index=False)
print(f"Kept   {after:,} / {before:,} PGK2-selective rows  ({100*after/before:.1f}%)")
print(f"Saved  → {out}")
