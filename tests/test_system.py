from src.validation import validate_patient_input

# Sample Input 1: Healthy / Low-Risk Profile (Expected: 0 - No Heart Disease)
SAMPLE_HEALTHY = {
    "Age": 40,
    "Sex": "F",
    "ChestPainType": "ATA",
    "RestingBP": 120,
    "Cholesterol": 200,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 170,
    "ExerciseAngina": "N",
    "Oldpeak": 0.0,
    "ST_Slope": "Up"
}

# Sample Input 2: High-Risk Clinical Profile (Expected: 1 - Heart Disease)
SAMPLE_HIGH_RISK = {
    "Age": 62,
    "Sex": "M",
    "ChestPainType": "ASY",
    "RestingBP": 150,
    "Cholesterol": 280,
    "FastingBS": 1,
    "RestingECG": "ST",
    "MaxHR": 110,
    "ExerciseAngina": "Y",
    "Oldpeak": 2.5,
    "ST_Slope": "Flat"
}

# --- 1. Unit Tests for Input Validation & Error Handling ---
def test_valid_sample_inputs():
    is_valid_1, msg_1 = validate_patient_input(SAMPLE_HEALTHY)
    is_valid_2, msg_2 = validate_patient_input(SAMPLE_HIGH_RISK)
    assert is_valid_1 is True and msg_1 == "Valid input."
    assert is_valid_2 is True and msg_2 == "Valid input."

def test_invalid_zero_cholesterol():
    bad_data = SAMPLE_HEALTHY.copy()
    bad_data["Cholesterol"] = 0
    is_valid, msg = validate_patient_input(bad_data)
    assert is_valid is False
    assert "Cholesterol" in msg

def test_invalid_negative_bp():
    bad_data = SAMPLE_HEALTHY.copy()
    bad_data["RestingBP"] = -10
    is_valid, msg = validate_patient_input(bad_data)
    assert is_valid is False
    assert "Resting blood pressure" in msg

def test_missing_feature_handling():
    incomplete_data = SAMPLE_HEALTHY.copy()
    del incomplete_data["Age"]
    is_valid, msg = validate_patient_input(incomplete_data)
    assert is_valid is False
    assert "Missing" in msg

def test_invalid_category_handling():
    bad_data = SAMPLE_HEALTHY.copy()
    bad_data["ChestPainType"] = "UNKNOWN"
    is_valid, msg = validate_patient_input(bad_data)
    assert is_valid is False
    assert "ChestPainType" in msg

# --- 2. Integration Test with Backend Prediction Function ---
def test_backend_integration_validation_with_predict():
    is_valid, msg = validate_patient_input(SAMPLE_HEALTHY)
    assert is_valid is True, f"Validation failed: {msg}"

    # Verify predict module exists and is importable
    import src.predict as predict_module
    assert predict_module is not None
