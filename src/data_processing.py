import json
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "Age", "Sex", "ChestPainType", "RestingBP",
    "Cholesterol", "FastingBS", "RestingECG",
    "MaxHR", "ExerciseAngina", "Oldpeak",
    "ST_Slope", "HeartDisease"
]


def load_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(path)
    print(f"Loaded shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def validate_data(df: pd.DataFrame) -> dict:
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if not df["HeartDisease"].isin([0, 1]).all():
        raise ValueError("HeartDisease must contain only 0 and 1")

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": {k: int(v) for k, v in df.isnull().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "invalid_age_count": int((df["Age"] <= 0).sum()),
        "zero_or_negative_bp": int((df["RestingBP"] <= 0).sum()),
        "zero_cholesterol": int((df["Cholesterol"] == 0).sum()),
    }


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cleaned = df.copy()
    audit = {
        "initial_rows": int(len(cleaned)),
        "duplicates_removed": 0,
        "invalid_age_rows_removed": 0,
        "resting_bp_set_to_nan": 0,
        "cholesterol_zero_set_to_nan": 0,
        "final_rows": 0,
    }

    dup_count = int(cleaned.duplicated().sum())
    if dup_count > 0:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    audit["duplicates_removed"] = dup_count

    invalid_age_mask = cleaned["Age"] <= 0
    audit["invalid_age_rows_removed"] = int(invalid_age_mask.sum())
    cleaned = cleaned[~invalid_age_mask].reset_index(drop=True)

    bp_zero_mask = cleaned["RestingBP"] <= 0
    audit["resting_bp_set_to_nan"] = int(bp_zero_mask.sum())
    cleaned.loc[bp_zero_mask, "RestingBP"] = pd.NA

    chol_zero_mask = cleaned["Cholesterol"] == 0
    audit["cholesterol_zero_set_to_nan"] = int(chol_zero_mask.sum())
    cleaned.loc[chol_zero_mask, "Cholesterol"] = pd.NA

    audit["final_rows"] = int(len(cleaned))
    return cleaned, audit


def generate_summary(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    validation: dict,
    cleaning_audit: dict,
) -> dict:
    return {
        "raw_shape": list(raw_df.shape),
        "cleaned_shape": list(cleaned_df.shape),
        "target_distribution": {
            str(k): int(v) for k, v in cleaned_df["HeartDisease"].value_counts().items()
        },
        "validation_before_cleaning": validation,
        "cleaning_actions": cleaning_audit,
        "missing_after_cleaning": {
            k: int(v) for k, v in cleaned_df.isnull().sum().items()
        },
    }


def export_data(
    cleaned_df: pd.DataFrame,
    summary: dict,
    output_csv: str,
    output_json: str,
) -> None:
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(output_csv, index=False)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
