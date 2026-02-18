import pandas as pd
import os
# -------- CONFIG --------
dataset_path = "C:/Users/maria/OneDrive - Maastricht University/Maria Diez Perez/datasets/"
INPUT_CSV = os.path.join(dataset_path, 'icu_dka_dataset_20260210.csv')
OUTPUT_CSV = os.path.join(dataset_path, 'icu_dka_dataset_simplify.csv')
RACE_COLUMN = "race"
NEW_COLUMN = "race"
# ------------------------


def simplify_race(race):
    if pd.isna(race):
        return "Unknown"

    race = race.strip().upper()

    # White
    if race.startswith("WHITE") or race == "PORTUGUESE":
        return "White"

    # Black
    if race.startswith("BLACK"):
        return "Black"

    # Asian
    if race.startswith("ASIAN"):
        return "Asian"

    # Hispanic / Latino (ethnicity-based)
    if race.startswith("HISPANIC") or race.startswith("LATINO"):
        return "Hispanic"

    # Explicit unknown / missing
    if race in {
        "UNKNOWN",
        "UNABLE TO OBTAIN",
        "PATIENT DECLINED TO ANSWER"
    }:
        return "Unknown"

    # Everything else
    return "Other"


def main():
    print("INPUT_CSV =", INPUT_CSV, "type =", type(INPUT_CSV))

    df = pd.read_csv(INPUT_CSV)

    if RACE_COLUMN not in df.columns:
        raise ValueError(f"Column '{RACE_COLUMN}' not found in CSV")

    df[NEW_COLUMN] = df[RACE_COLUMN].apply(simplify_race)

    df.to_csv(OUTPUT_CSV, index=False)

    print("Race simplification complete.")
    print(df[NEW_COLUMN].value_counts(dropna=False))


if __name__ == "__main__":
    main()
