# Columns that must be >= 0
NON_NEGATIVE_COLUMNS = [
    "los",
    "ph_initial", "ph_min", "ph_max",
    "calcium_initial", "calcium_min", "calcium_max",
    "creatinine_initial", "creatinine_min", "creatinine_max",
    "hemoglobin_initial", "hemoglobin_min", "hemoglobin_max",
    "wbc_initial", "wbc_min", "wbc_max",
    "lactate_initial", "lactate_min", "lactate_max",
    "platelet_initial", "platelet_min", "platelet_max",
    "potassium_initial", "potassium_min", "potassium_max",
    "sodium_initial", "sodium_min", "sodium_max",
    "glucose_initial", "glucose_min", "glucose_max",
    "chloride_initial", "chloride_min", "chloride_max",
    "bun_initial", "bun_min", "bun_max",
    "osmolality_initial", "osmolality_min", "osmolality_max",
    "anion_gap_initial", "anion_gap_min", "anion_gap_max",
    "bicarbonate_initial", "bicarbonate_min", "bicarbonate_max",
    "heart_rate_initial", "heart_rate_max",
    "mbp_initial", "mbp_min", "mbp_max",
    "resp_rate_initial", "resp_rate_max",
    "urineoutput",
]

# Columns that must lie in [min, max] (None means open-ended)
RANGE_CONSTRAINTS = {
    # demographics
    "age": (0, 100),
    "weight": (2, 300),  # kg

    # scores
    "sofa": (0, 24),
    "gcs": (3, 15),
    "apsiii": (0, 200),
    "cns": (0, 12),

    # vitals – broad physiological ranges
    # "heart_rate_initial": (20, 250),
    # "heart_rate_max": (20, 250),
    # "mbp_initial": (20, 200),
    # "mbp_min": (20, 200),
    # "mbp_max": (20, 200),
    # "temperature_max": (30, 45),
    # "resp_rate_initial": (5, 80),
    # "resp_rate_max": (5, 80),

    # # pH – generous physiological bounds
    # "ph_initial": (6.5, 8.0),
    # "ph_min": (6.5, 8.0),
    # "ph_max": (6.5, 8.0),
    
}
BINARY_COLUMNS = [
    "aki",
    "ckd",
    "diabetes_with_cc",
    "diabetes_without_cc",
    "cardiac_disease",
    "hypertension",
    "copd",
    "renal",
    "ventilation",
    "nitroprusside",
    "vasopressor",
    "albumin",
    "in_hospital_death",
    "emergency",
]
