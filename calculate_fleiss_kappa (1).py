#!/usr/bin/env python3
"""
calculate_fleiss_kappa.py
Read a CSV with rater labels and compute Fleiss' Kappa for nominal data with >=2 raters.
This version has NO dependency on statsmodels (works on Python 3.8+).

Expected CSV:
- Contains at least 3 columns with rater labels (strings). You can specify them with --raters,
  otherwise the script will try to auto-detect the last 3 non-empty columns as raters.
- Each rater cell must contain exactly one label (rows with missing values or commas are excluded).

Usage:
    python calculate_fleiss_kappa.py ratings.csv --raters "Label Max" "Label Guusje" "Label Matthias"
    # or rely on auto-detection:
    python calculate_fleiss_kappa.py ratings.csv
"""

import argparse
import sys
from typing import List, Tuple
import pandas as pd
import numpy as np


def build_count_matrix(df: pd.DataFrame, rater_cols: List[str]) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Return (count_matrix, label_list, clean_df).
    count_matrix shape: (N_items, K_labels) with counts per item per label.
    Rows with NaN or multi-label entries (comma-separated) are dropped.
    """
    labels_df = df[rater_cols].copy()
    clean_df = labels_df.dropna()
    # Exclude rows where any rater gave multiple labels (commas)
    clean_df = clean_df[~clean_df.apply(lambda row: row.astype(str).str.contains(",").any(), axis=1)]
    if clean_df.empty:
        raise ValueError("No valid rows after cleaning (missing values or multiple labels).")

    # Collect unique labels
    unique_labels = sorted(set(clean_df.values.flatten()))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}

    # Build matrix
    N = len(clean_df)
    K = len(unique_labels)
    M = np.zeros((N, K), dtype=int)
    for i, row in enumerate(clean_df.values):
        for lab in row:
            M[i, label_to_idx[str(lab)]] += 1
    return M, unique_labels, clean_df


def fleiss_kappa_from_counts(M: np.ndarray) -> float:
    """
    Compute Fleiss' Kappa from a count matrix M (N_items x K_labels),
    where each row sums to n (the number of raters per item).
    Formula per Fleiss (1971).
    """
    if M.size == 0:
        raise ValueError("Empty matrix.")
    n_per_item = M.sum(axis=1)
    if not np.all(n_per_item == n_per_item[0]):
        raise ValueError("All items must have the same number of ratings; found variation.")
    n = int(n_per_item[0])
    if n < 2:
        raise ValueError("Need at least 2 raters per item.")

    N, K = M.shape
    # P_i: agreement for item i
    P_i = ( (M**2).sum(axis=1) - n ) / ( n * (n - 1) )
    P_bar = P_i.mean()

    # p_j: overall proportion for each category
    p_j = M.sum(axis=0) / (N * n)
    P_e_bar = (p_j ** 2).sum()

    if np.isclose(1 - P_e_bar, 0):
        return float("nan")
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    return float(kappa)


def autodetect_rater_cols(df: pd.DataFrame, n: int = 3) -> List[str]:
    """Pick the last n non-empty columns as rater columns (common pattern)."""
    non_empty_cols = [c for c in df.columns if not df[c].dropna().empty]
    if len(non_empty_cols) < n:
        raise ValueError(f"Not enough non-empty columns to auto-detect {n} rater columns.")
    return non_empty_cols[-n:]


def main():
    parser = argparse.ArgumentParser(description="Calculate Fleiss' Kappa from CSV ratings (no statsmodels dependency).")
    parser.add_argument("csv_path", help="Path to CSV file containing rater labels.")
    parser.add_argument("--raters", nargs="+", help="Column names of rater labels (space-separated).")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        sys.exit(f"Error reading CSV: {e}")

    # Determine rater columns
    if args.raters:
        rater_cols = args.raters
        for col in rater_cols:
            if col not in df.columns:
                sys.exit(f"Rater column not found in CSV: '{col}'. Available columns: {list(df.columns)}")
    else:
        rater_cols = autodetect_rater_cols(df, n=3)

    try:
        M, labels, clean_df = build_count_matrix(df, rater_cols)
        kappa = fleiss_kappa_from_counts(M)
    except Exception as e:
        sys.exit(f"Error computing Fleiss' Kappa: {e}")

    print(f"Rater columns: {rater_cols}")
    print(f"Labels found: {labels}")
    print(f"Items included (after cleaning): {len(clean_df)}")
    print(f"Fleiss' Kappa: {kappa:.3f}")


if __name__ == "__main__":
    main()
