import pandas as pd

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

