"""
merge_results.py
~~~~~~~~~~~~~~~~
Merges all scattered per-model / per-N CSV files under
  src/多模型预测结果/global/       → metrics_global_<model>.csv
  src/多模型预测结果/comparison/   → comparison_<model>_M<m>_N<n>.csv

into two unified CSV files:
  src/多模型预测结果/merged_global_metrics.csv
  src/多模型预测结果/merged_comparison_metrics.csv

Run from the repository root:
  python src/merge_results.py
"""

import os
import glob
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "多模型预测结果")
GLOBAL_DIR = os.path.join(BASE, "global")
COMP_DIR = os.path.join(BASE, "comparison")
OUT_GLOBAL = os.path.join(BASE, "merged_global_metrics.csv")
OUT_COMP = os.path.join(BASE, "merged_comparison_metrics.csv")


# ── helpers ────────────────────────────────────────────────────────────────────

def _parse_model_M_N(filename: str):
    """
    Extract model, M, N from filenames like:
      comparison_random_forest_M60_N105.csv
    Returns (model_name, M_int, N_int).
    """
    stem = os.path.basename(filename).replace("comparison_", "").replace(".csv", "")
    parts = stem.split("_")
    model_parts, m_val, n_val = [], None, None
    for p in parts:
        if p.startswith("M") and p[1:].isdigit():
            m_val = int(p[1:])
        elif p.startswith("N") and p[1:].isdigit():
            n_val = int(p[1:])
        else:
            model_parts.append(p)
    return "_".join(model_parts), m_val, n_val


# ── 1. Merge global metrics ────────────────────────────────────────────────────

def merge_global():
    frames = []
    for f in glob.glob(os.path.join(GLOBAL_DIR, "metrics_global_*.csv")):
        # Some files have an extra trailing column (e.g., NRMSE) appended after R2
        # without a corresponding header entry, which causes a C-engine parse error.
        # Strategy: read the actual header first to determine the declared column count,
        # then re-read with one extra overflow slot so all rows are parsed correctly.
        with open(f, encoding="utf-8-sig") as fh:
            header_cols = fh.readline().strip().split(",")
        # Always allocate one extra slot beyond the declared header so that rows with
        # a trailing value do not trigger "unexpected fields" errors.
        padded_names = header_cols + ["_extra"]
        df = pd.read_csv(f, encoding="utf-8-sig", names=padded_names, skiprows=1)
        # Validate that expected base columns are present; warn if not.
        base_cols = ["model", "mode", "M", "N", "experiment_id", "RMSE", "MAE", "R2"]
        missing = [c for c in base_cols if c not in df.columns]
        if missing:
            print(f"  WARNING: {os.path.basename(f)} is missing columns {missing}, skipping.")
            continue
        frames.append(df[base_cols])

    merged = pd.concat(frames, ignore_index=True)

    # De-duplicate: for the same (model, M, N) keep the row with the latest
    # experiment_id. The IDs are zero-padded UTC timestamps (YYYYMMDD_HHMMSS) so
    # lexicographic ordering is equivalent to chronological ordering.
    merged = (
        merged
        .sort_values("experiment_id", ascending=True)
        .drop_duplicates(subset=["model", "M", "N"], keep="last")
        .sort_values(["model", "N"])
        .reset_index(drop=True)
    )

    merged.to_csv(OUT_GLOBAL, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(merged)} rows → {OUT_GLOBAL}")
    return merged


# ── 2. Merge comparison metrics ────────────────────────────────────────────────

def merge_comparison():
    rows = []
    for f in glob.glob(os.path.join(COMP_DIR, "comparison_*.csv")):
        model_name, m_val, n_val = _parse_model_M_N(f)
        df = pd.read_csv(f, encoding="utf-8-sig")
        for _, row in df.iterrows():
            entry = {
                "model": model_name,
                "M": m_val,
                "N": n_val,
                "label": row["label"],
                "RMSE": row["RMSE"],
                "MAE": row["MAE"],
                "R2": row["R2"],
            }
            if "NRMSE" in df.columns:
                entry["NRMSE"] = row["NRMSE"]
            rows.append(entry)

    merged = (
        pd.DataFrame(rows)
        .sort_values(["model", "N", "label"])
        .reset_index(drop=True)
    )

    # Reorder columns: model, M, N, label, RMSE, MAE, R2, NRMSE
    col_order = ["model", "M", "N", "label", "RMSE", "MAE", "R2"]
    if "NRMSE" in merged.columns:
        col_order.append("NRMSE")
    merged = merged[col_order]

    merged.to_csv(OUT_COMP, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(merged)} rows → {OUT_COMP}")
    return merged


if __name__ == "__main__":
    g = merge_global()
    print(g.to_string(index=False))
    print()
    c = merge_comparison()
    print(c.to_string(index=False))
