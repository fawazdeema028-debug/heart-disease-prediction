REQUIRED_FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
]

VALID_CATEGORIES = {
    "Sex": ["M", "F"],
    "ChestPainType": ["TA", "ATA", "NAP", "ASY"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["Y", "N"],
    "ST_Slope": ["Up", "Flat", "Down"]
}

def validate_patient_input(data: dict) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Input data must be a dictionary."

    missing = [feat for feat in REQUIRED_FEATURES if feat not in data or data[feat] is None]
    if missing:
        return False, f"Missing required features: {', '.join(missing)}"

    try:
        age = float(data["Age"])
        resting_bp = float(data["RestingBP"])
        cholesterol = float(data["Cholesterol"])
        fasting_bs = int(data["FastingBS"])
        max_hr = float(data["MaxHR"])
        oldpeak = float(data["Oldpeak"])
    except (ValueError, TypeError):
        return False, "Numerical features must contain valid numbers."

    if not (1 <= age <= 120):
        return False, "Age must be between 1 and 120."
    if resting_bp <= 0:
        return False, "Resting blood pressure must be positive."
    if cholesterol <= 0:
        return False, "Cholesterol must be positive."
    if fasting_bs not in (0, 1):
        return False, "FastingBS must be 0 or 1."
    if not (40 <= max_hr <= 250):
        return False, "MaxHR must be between 40 and 250."
    if not (-5.0 <= oldpeak <= 10.0):
        return False, "Oldpeak value is out of valid clinical range."

    for col, allowed in VALID_CATEGORIES.items():
        if data[col] not in allowed:
            return False, f"Invalid value for {col}: '{data[col]}'. Expected one of {allowed}."

    return True, "Valid input."
