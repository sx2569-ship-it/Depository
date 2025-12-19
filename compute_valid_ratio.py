from __future__ import annotations
import argparse
import os
from typing import Dict
import cv2
import numpy as np
import pandas as pd

DEFAULT_MAPPING: Dict[str, str] = {
    'ORB + FEATHER': 'orb_feather_mask',
    'ORB + SEAMMB': 'orb_seammb_mask',
    'SIFT + FEATHER': 'sift_feather_mask',
    'SIFT + SEAMMB': 'sift_seammb_mask',
}

def valid_ratio_from_mask(mask_path: str, thresh: int = 1) -> float:
    """Return fraction of pixels >= thresh."""
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(mask_path)
    return float((m >= thresh).mean())

def compute_valid_ratio_for_results(results_dir: str, df: pd.DataFrame, mapping: Dict[str, str] = None) -> pd.DataFrame:
    """Compute and merge valid ratio into an existing metrics DataFrame."""
    mapping = mapping or DEFAULT_MAPPING
    rows = []
    for setting, mask_fn in mapping.items():
        path = None
        for ext in ('.png', '.jpg', '.jpeg'):
            cand = os.path.join(results_dir, mask_fn + ext)
            if os.path.exists(cand):
                path = cand
                break
        if path is None:
            raise FileNotFoundError(f'No mask found for {setting}: expected {mask_fn}.png/.jpg/.jpeg in {results_dir}')
        rows.append((setting, valid_ratio_from_mask(path)))
    df_ratio = pd.DataFrame(rows, columns=["Setting", "Valid Ratio"])
    out = df.merge(df_ratio, on="Setting", how="left")
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--csv", type=str, default="metrics_summary.csv")
    args = parser.parse_args()
    csv_path = os.path.join(args.results_dir, args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    out = compute_valid_ratio_for_results(args.results_dir, df)
    out_path = os.path.join(args.results_dir, "metrics_summary_with_valid_ratio.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()
