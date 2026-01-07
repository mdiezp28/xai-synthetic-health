import pandas as pd

from .clinical_rules import NON_NEGATIVE_COLUMNS, RANGE_CONSTRAINTS, BINARY_COLUMNS

def post_process_synthetic_data(df):
    """
    Post-process synthetic data using clinical rules and real data ranges:
    1. Drop rows that violate non-negative constraints
    2. Drop rows that violate range constraints from clinical rules
    3. Drop rows that violate binary constraints
    4. Enforce min/max ordering (e.g., min <= max) by swapping if needed
    """
    df_processed = df.copy()

    # 1. Drop rows with negative values in non-negative columns
    valid_mask = pd.Series(True, index=df_processed.index)
    for col in NON_NEGATIVE_COLUMNS:
        if col in df_processed.columns:
            valid_mask &= df_processed[col].ge(0)
    
    # 2. Drop rows outside range constraints from clinical rules
    for col, (low, high) in RANGE_CONSTRAINTS.items():
        if col in df_processed.columns:
            valid_mask &= df_processed[col].ge(low) & df_processed[col].le(high)
    
    # 3. Drop rows that violate binary constraints
    for col in BINARY_COLUMNS:
        if col in df_processed.columns:
            valid_mask &= df_processed[col].isin([0, 1])

    df_processed = df_processed[valid_mask].copy()
    
    # 4. Enforce min/max ordering constraints
    min_max_pairs = [
        ('ph_min', 'ph_max'),
        ('calcium_min', 'calcium_max'),
        ('creatinine_min', 'creatinine_max'),
        ('hemoglobin_min', 'hemoglobin_max'),
        ('wbc_min', 'wbc_max'),
        ('lactate_min', 'lactate_max'),
        ('platelet_min', 'platelet_max'),
        ('potassium_min', 'potassium_max'),
        ('sodium_min', 'sodium_max'),
        ('glucose_min', 'glucose_max'),
        ('chloride_min', 'chloride_max'),
        ('bun_min', 'bun_max'),
        ('osmolality_min', 'osmolality_max'),
        ('anion_gap_min', 'anion_gap_max'),
        ('bicarbonate_min', 'bicarbonate_max'),
        ('mbp_min', 'mbp_max'),
    ]
    
    for min_col, max_col in min_max_pairs:
        if min_col in df_processed.columns and max_col in df_processed.columns:
            # Swap if min > max
            mask = df_processed[min_col] > df_processed[max_col]
            df_processed.loc[mask, [min_col, max_col]] = \
                df_processed.loc[mask, [max_col, min_col]].values
    
    return df_processed

# syn_data = pd.read_csv("./dp_cgans/tests/output/2026_01_02_19_34_39_syn_data_10f_lr_5e-5.csv")
# post_syn = post_process_synthetic_data(syn_data)
# post_syn.to_csv("./dp_cgans/tests/output/syn_data_10f_lr_5e-5_postprocessed.csv", index=False)