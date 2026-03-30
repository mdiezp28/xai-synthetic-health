import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path("C:/Users/Maria/iCloudDrive/Documents/Studies/AI/Thesis/full_results/utility/")     
DATANAME = "baseline"
REAL_DIR = BASE_DIR / "real"

EXP_GLOB = "exp*"

# If feature sets differ:
#   "raise" => stop at first mismatch after reporting
#   "skip"  => skip that fold and continue
ON_MISMATCH = "raise"

OUT_DIR = BASE_DIR / "_spearman_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_OUT = OUT_DIR / "all_experiments_vs_real_spearman.csv"
SUMMARY_OUT = OUT_DIR / "summary_spearman.csv"

MISMATCH_DIR = OUT_DIR / "feature_mismatches"
MISMATCH_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def load_feature_importance(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)

    required = {"feature", "importance"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns {required}. Found: {list(df.columns)}")

    df = df[["feature", "importance"]].copy()
    df["feature"] = df["feature"].astype(str)
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    df = df.dropna(subset=["importance"])

    # Aggregate duplicates (sum). If you prefer mean: .mean()
    return df.groupby("feature", as_index=True)["importance"].sum()


def fold_id_from_path(p: Path) -> str:
    m = re.search(r"(fold_\d+)", str(p))
    return m.group(1) if m else "unknown_fold"


def save_feature_mismatch(dataname: str, exp_name: str, fold: str,
                          exp_s: pd.Series, real_s: pd.Series,
                          exp_path: Path, real_path: Path) -> Path:
    exp_feats = set(exp_s.index)
    real_feats = set(real_s.index)

    only_in_exp = sorted(exp_feats - real_feats)
    only_in_real = sorted(real_feats - exp_feats)

    out_csv = MISMATCH_DIR / f"{dataname}__{exp_name}__{fold}__feature_mismatch.csv"
    pd.DataFrame({
        "only_in_experiment": pd.Series(only_in_exp, dtype="string"),
        "only_in_real": pd.Series(only_in_real, dtype="string"),
    }).to_csv(out_csv, index=False)

    print(f"\n[FEATURE MISMATCH] dataname={dataname} exp={exp_name} fold={fold}")
    print(f"  exp file : {exp_path}")
    print(f"  real file: {real_path}")
    print(f"  exp features : {len(exp_feats)}")
    print(f"  real features: {len(real_feats)}")
    print(f"  only in exp  : {len(only_in_exp)}")
    print(f"  only in real : {len(only_in_real)}")
    print(f"  Saved mismatch list: {out_csv}")

    return out_csv


def spearman_on_same_features(exp_s: pd.Series, real_s: pd.Series) -> float:
    exp_al = exp_s.sort_index()
    real_al = real_s.reindex(exp_al.index)

    if len(exp_al) < 2:
        return np.nan
    rho, _ = spearmanr(exp_al.values, real_al.values)
    return float(rho)


def upsert_csv(path: Path, new_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """
    Update existing rows by key_cols and append new rows.
    Keeps the last occurrence per key (new_df overrides old).
    """
    if path.exists():
        old_df = pd.read_csv(path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = new_df.copy()

    # nice stable ordering if present
    for col in ["dataname", "experiment", "fold"]:
        if col in combined.columns:
            pass

    combined.to_csv(path, index=False)
    return combined


# ----------------------------
# Main
# ----------------------------
def main():
    dataname_dir = BASE_DIR / DATANAME
    exp_dirs = sorted([p for p in dataname_dir.glob(EXP_GLOB) if p.is_dir()])
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment directories found at: {dataname_dir}/{EXP_GLOB}")

    # Collect fold-level rows for this DATANAME run
    fold_rows = []

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name

        exp_pattern = str(exp_dir / "fold_*" / f"{exp_name}_fold_*_feature_importance_shap_grouped.csv")
        exp_files = sorted(glob.glob(exp_pattern))
        if not exp_files:
            print(f"[WARN] No fold CSVs found for {exp_name} with pattern:\n  {exp_pattern}")
            continue

        for exp_file in exp_files:
            exp_path = Path(exp_file)
            fold = fold_id_from_path(exp_path)

            # ✅ real file name per your note:
            # BASE_DIR/real/fold_i/fold_i_feature_importance_shap_grouped.csv
            real_path = REAL_DIR / fold / f"{fold}_feature_importance_shap_grouped.csv"
            if not real_path.exists():
                print(f"[WARN] Missing real file for fold {fold}: {real_path}")
                continue

            exp_s = load_feature_importance(exp_path)
            real_s = load_feature_importance(real_path)

            if set(exp_s.index) != set(real_s.index):
                save_feature_mismatch(DATANAME, exp_name, fold, exp_s, real_s, exp_path, real_path)
                if ON_MISMATCH == "raise":
                    raise ValueError(
                        f"Feature sets differ for dataname={DATANAME} exp={exp_name} fold={fold}. "
                        f"See mismatch CSVs in {MISMATCH_DIR}/"
                    )
                elif ON_MISMATCH == "skip":
                    continue
                else:
                    raise ValueError("ON_MISMATCH must be 'raise' or 'skip'.")

            rho = spearman_on_same_features(exp_s, real_s)

            fold_rows.append({
                "dataname": DATANAME,
                "experiment": exp_name,
                "fold": fold,
                "n_features": int(len(exp_s)),
                "spearman_rho": rho,
                "exp_file": str(exp_path),
                "real_file": str(real_path)
            })

    if not fold_rows:
        print("No fold-level results found (nothing to write).")
        return

    df_new = pd.DataFrame(fold_rows).sort_values(["experiment", "fold"])

    # 1) Upsert fold-level file
    df_all = upsert_csv(
        ALL_OUT,
        df_new,
        key_cols=["dataname", "experiment", "fold"]
    )

    # 2) Build per-experiment summary for THIS run (DATANAME only)
    df_summary_new = (
        df_new.dropna(subset=["spearman_rho"])
             .groupby(["dataname", "experiment"], as_index=False)
             .agg(
                 mean_rho=("spearman_rho", "mean"),
                 median_rho=("spearman_rho", "median"),
                 n_folds=("spearman_rho", "count"),
                 n_features=("n_features", "first"),
             )
    )
    # Upsert summary across multiple datanames
    df_summary_all = upsert_csv(
        SUMMARY_OUT,
        df_summary_new,
        key_cols=["dataname", "experiment"]
    )

    # Optional: show quick console summary for this run
    print(f"\nWrote/updated: {ALL_OUT}")
    print(f"Wrote/updated: {SUMMARY_OUT}")
    print("\nThis run (per-experiment):")
    print(df_summary_new.sort_values(["experiment"])[["dataname", "experiment", "mean_rho", "median_rho", "n_folds"]].to_string(index=False))


if __name__ == "__main__":
    main()