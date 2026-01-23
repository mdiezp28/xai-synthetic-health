import numpy as np

from syn_data_evaluation.data.clinical_rules import NON_NEGATIVE_COLUMNS, RANGE_CONSTRAINTS, BINARY_COLUMNS

SCORE_COLS = ["apsiii", "gcs", "cns", "renal"]

MIN_INIT_MAX_TRIPLES = [
    ("ph_min", "ph_initial", "ph_max"),
    ("calcium_min", "calcium_initial", "calcium_max"),
    ("creatinine_min", "creatinine_initial", "creatinine_max"),
    ("hemoglobin_min", "hemoglobin_initial", "hemoglobin_max"),
    ("wbc_min", "wbc_initial", "wbc_max"),
    ("lactate_min", "lactate_initial", "lactate_max"),
    ("platelet_min", "platelet_initial", "platelet_max"),
    ("potassium_min", "potassium_initial", "potassium_max"),
    ("sodium_min", "sodium_initial", "sodium_max"),
    ("glucose_min", "glucose_initial", "glucose_max"),
    ("chloride_min", "chloride_initial", "chloride_max"),
    ("bun_min", "bun_initial", "bun_max"),
    ("osmolality_min", "osmolality_initial", "osmolality_max"),
    ("anion_gap_min", "anion_gap_initial", "anion_gap_max"),
    ("bicarbonate_min", "bicarbonate_initial", "bicarbonate_max"),
    ("mbp_min", "mbp_initial", "mbp_max"),
]

def _postprocess_synthetic_data(df, is_utility=False):
    df = df.copy()
    # inf nan to NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 1) Binaries: threshold, keep NaNs
    for c in BINARY_COLUMNS:
        if c in df.columns:
            mask = df[c].notna()
            df.loc[mask, c] = (df.loc[mask, c] >= 0.5).astype(int)

    # 2) Scores: round only (no clip), keep NaNs
    for c in SCORE_COLS:
        if c in df.columns:
            df[c] = df[c].round()

    # 3) Enforce min<=max and min<=initial<=max
    for mn, init, mx in MIN_INIT_MAX_TRIPLES:
        df = _enforce_min_init_max_relabel(df, mn, init, mx)

    if is_utility:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            if c not in BINARY_COLUMNS and c not in SCORE_COLS:
                df[c] = df[c].round(2)

        # Non-negative columns: clip at 0 
        for c in NON_NEGATIVE_COLUMNS:
            if c in df.columns:
                m = df[c].notna()
                df.loc[m, c] = df.loc[m, c].clip(lower=0)
        # Range constraints: 
        for c, (lo, hi) in RANGE_CONSTRAINTS.items():
            if c in df.columns:
                m = df[c].notna()
                if lo is not None:
                    df.loc[m, c] = df.loc[m, c].clip(lower=lo)
                if hi is not None:
                    df.loc[m, c] = df.loc[m, c].clip(upper=hi)
    # print(f"Post-processing complete. Data shape: {df.shape}")
    return df

def _enforce_min_init_max_relabel(df, mn, init, mx):
    df = df.copy()
    cols = [mn, init, mx]
    if not all(c in df.columns for c in cols):
        return df

    # work row-wise only where all three exist
    mask = df[mn].notna() & df[init].notna() & df[mx].notna()
    vals = df.loc[mask, cols].to_numpy()

    # sort each row: smallest->mn, middle->init, largest->mx
    vals_sorted = np.sort(vals, axis=1)

    df.loc[mask, mn] = vals_sorted[:, 0]
    df.loc[mask, init] = vals_sorted[:, 1]
    df.loc[mask, mx] = vals_sorted[:, 2]
    return df


def postprocess_for_utility(df):
    return _postprocess_synthetic_data(df, is_utility=True)

def postprocess_for_fidelity(df):
    return _postprocess_synthetic_data(df, is_utility=False)
