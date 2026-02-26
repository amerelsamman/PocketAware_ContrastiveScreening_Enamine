#!/usr/bin/env python3
"""Download REINVENT4 prior models from Zenodo."""

import os
import sys
import requests
from pathlib import Path

ZENODO_API = "https://zenodo.org/api/records/15641296"
PRIORS_DIR = Path("reinvent4/priors")

def download_file(url, filename):
    """Download a file from URL and save it."""
    print(f"Downloading {filename}...")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        size_mb = os.path.getsize(filename) / (1024*1024)
        print(f"✓ Downloaded {filename} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def main():
    """Fetch prior models from Zenodo."""
    PRIORS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Fetching Zenodo record metadata...")
        resp = requests.get(ZENODO_API, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Failed to fetch Zenodo metadata: {e}")
        return False
    
    files = data.get('files', [])
    print(f"Found {len(files)} files in Zenodo record")
    
    # Look for .prior files
    prior_files = [f for f in files if f['filename'].endswith('.prior')]
    print(f"Found {len(prior_files)} prior files")
    
    if not prior_files:
        print("No .prior files found. Available files:")
        for f in files[:10]:
            print(f"  - {f['filename']}")
        return False
    
    # Download first prior (usually reinvent.prior)
    prior_file = prior_files[0]
    filename = prior_file['filename']
    download_url = prior_file['links']['self']
    
    target_path = PRIORS_DIR / filename
    if target_path.exists():
        size_mb = target_path.stat().st_size / (1024*1024)
        print(f"✓ {filename} already exists ({size_mb:.1f} MB)")
        return True
    
    return download_file(download_url, str(target_path))

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
