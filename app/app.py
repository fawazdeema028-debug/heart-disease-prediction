
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.predict import predict
# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Custom UI styling
st.markdown("""
<style>

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #D9EAE8;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(15, 118, 110, 0.08);
}

[data-testid="stMetricValue"] {
    color: #0F766E;
    font-weight: 700;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
}

.stButton button,
.stFormSubmitButton button {
    border-radius: 10px;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    background-color: #E7F3F2;
}

h1, h2, h3 {
    color: #12343B;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #D9EAE8;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(15, 118, 110, 0.06);
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] label {
    padding: 10px;
    border-radius: 10px;
}

[data-testid="stSidebar"] label:hover {
    background-color: #CDE8E5;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown(
    ":green[:material/ecg_heart:] **CardioAI**"
)
st.sidebar.caption("Heart Disease Prediction System")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction", "Data Visualization", "About"],
    format_func=lambda x: {
        "Home": "🏠  Home",
        "Prediction": "🩺  Prediction",
        "Data Visualization": "📊  Data Visualization",
        "About": "ℹ️  About"
    }[x],
    label_visibility="collapsed",
    key="navigation"
)

# Home page

if page == "Home":

    st.title(":green[:material/ecg_heart:] CardioAI")

    st.subheader("Intelligent Heart Disease Prediction")

    st.write(
        "A machine learning application designed to analyze "
        "clinical information and predict heart disease."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Patient Records", "918", border=True)
    col2.metric("Clinical Features", "11", border=True)
    col3.metric("Prediction Classes", "2", border=True)

    st.divider()

    st.subheader("How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.header("📋")
            st.markdown("**01. Enter Information**")
            st.write("Provide the required clinical measurements.")

    with col2:
        with st.container(border=True):
            st.header("⚙️")
            st.markdown("**02. AI Analysis**")
            st.write("The trained machine learning model analyzes the data.")

    with col3:
        with st.container(border=True):
            st.header("📊")
            st.markdown("**03. View Results**")
            st.write("Review the prediction and model confidence.")

    st.divider()
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Get Started")
            st.write(
                "Enter clinical information to generate "
                "an AI-based prediction."
            )

        with col2:
            st.button(
                "Go to Prediction →",
                type="primary",
                on_click=lambda: st.session_state.update(
                    navigation="Prediction"
                )
            )


# Prediction page

elif page == "Prediction":
    st.title("❤️ Heart Disease Prediction")
    st.write("Enter the patient's clinical information.")

    with st.form("prediction_form"):

        st.subheader("Personal Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age", min_value=1, max_value=120, value=50
            )

        with col2:
            sex = st.selectbox("Sex", ["M", "F"])

        st.divider()
        st.subheader("Clinical Measurements")

        col1, col2, col3 = st.columns(3)

        with col1:
            resting_bp = st.number_input(
                "Resting Blood Pressure (mmHg)",
                min_value=0, value=120
            )

            cholesterol = st.number_input(
                "Cholesterol (mg/dL)",
                min_value=0, value=200
            )

        with col2:
            fasting_bs = st.selectbox(
                "Fasting Blood Sugar", [0, 1],
                format_func=lambda x: "Above 120 mg/dL"
                if x == 1 else "120 mg/dL or below"
            )

            max_hr = st.number_input(
                "Maximum Heart Rate (bpm)",
                min_value=1, value=150
            )

        with col3:
            chest_pain = st.selectbox(
                "Chest Pain Type",
                ["TA", "ATA", "NAP", "ASY"]
            )

            resting_ecg = st.selectbox(
                "Resting ECG", ["Normal", "ST", "LVH"]
            )

        st.divider()
        st.subheader("Exercise Test Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            exercise_angina = st.selectbox(
                "Exercise-Induced Angina", ["N", "Y"]
            )

        with col2:
            oldpeak = st.number_input(
                "Oldpeak", value=0.0, step=0.1
            )

        with col3:
            st_slope = st.selectbox(
                "ST Slope", ["Up", "Flat", "Down"]
            )

        submitted = st.form_submit_button(
            "Predict Heart Disease", type="primary"
        )

    
    if submitted:
        if resting_bp <= 0 or cholesterol <= 0:
            st.error(
                "Blood pressure and cholesterol must be positive."
            )

        else:
            user_data = {
                "Age": age,
                "Sex": sex,
                "ChestPainType": chest_pain,
                "RestingBP": resting_bp,
                "Cholesterol": cholesterol,
                "FastingBS": fasting_bs,
                "RestingECG": resting_ecg,
                "MaxHR": max_hr,
                "ExerciseAngina": exercise_angina,
                "Oldpeak": oldpeak,
                "ST_Slope": st_slope
            }

            try:
                with st.spinner("Analyzing clinical data..."):
                    result = predict(user_data)

                prediction = result["prediction"]
                confidence = result["confidence"]

                st.divider()
                st.subheader("Prediction Results")

                col1, col2 = st.columns(2)

                with col1:
                    if prediction == 1:
                        st.warning("Heart Disease Predicted")
                    else:
                        st.success("No Heart Disease Predicted")

                with col2:
                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                st.progress(confidence / 100)

                st.caption(
                    "Model confidence is not a guarantee "
                    "of clinical accuracy."
                )

            except Exception as e:
                st.error(
                    "Prediction failed. Please check the inputs "
                    "and model configuration."
                )
                with st.expander("Technical Details"):
                    st.code(str(e))


# Data visualization page

elif page == "Data Visualization":
    st.title("📊 Data Visualization")

    st.write(
        "Explore the heart disease dataset through "
        "interactive statistics and clinical visualizations."
    )

    data_path = ROOT_DIR / "data" / "processed" / "heart_cleaned.csv"
    figures_path = ROOT_DIR / "outputs" / "figures"

    try:
        df = pd.read_csv(data_path)

        # Dataset overview
        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", len(df))

        col2.metric(
            "Heart Disease Cases",
            int((df["HeartDisease"] == 1).sum())
        )

        col3.metric(
            "No Heart Disease Cases",
            int((df["HeartDisease"] == 0).sum())
        )

        st.divider()

        # Visualization selection
        st.subheader("Explore Clinical Data")

        charts = {
            "Heart Disease Distribution": (
                "target_dist.png",
                "Shows the distribution of patients with and "
                "without heart disease."
            ),
            "Age Distribution": (
                "age_dist.png",
                "Shows the distribution of patient ages "
                "according to heart disease status."
            ),
            "Chest Pain Analysis": (
                "chest_pain.png",
                "Compares chest pain types and "
                "heart disease status."
            ),
            "Cholesterol Distribution": (
                "cholesterol_box.png",
                "Shows cholesterol levels and their "
                "distribution across both classes."
            ),
            "Correlation Heatmap": (
                "correlation_heatmap.png",
                "Displays correlations between numerical "
                "clinical features and heart disease."
            )
        }

        selected_chart = st.selectbox(
            "Select a Visualization",
            list(charts.keys())
        )

        filename, description = charts[selected_chart]

        st.write(description)

        image_path = figures_path / filename

        if image_path.exists():
            st.image(str(image_path), width=700)
        else:
            st.warning("The selected visualization is unavailable.")

        st.divider()

        # Dataset preview
        with st.expander("View Dataset Sample"):
            st.dataframe(df.head(20))

    except Exception as e:
        st.error("Unable to load the dataset or visualizations.")

        with st.expander("Technical Details"):
            st.code(str(e))


# About page

elif page == "About":
    st.title("About CardioAI")

    st.subheader("Project Overview")
    st.write(
        "CardioAI is an academic machine learning application "
        "developed to predict heart disease using clinical data."
    )

    st.divider()

    st.subheader("Target Users")
    st.write(
        "Healthcare professionals, researchers, and students "
        "exploring AI-based clinical decision support."
    )

    st.divider()

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Patient Records", "918", border=True)
    col2.metric("Input Features", "11", border=True)
    col3.metric("Target Classes", "2", border=True)

    st.write("Source: Kaggle Heart Failure Prediction Dataset.")

    st.divider()

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas and NumPy
    - Scikit-learn
    - Matplotlib and Seaborn
    - Joblib
    """)

    st.divider()

    st.subheader("System Limitations")

    st.markdown("""
    - The model is trained on a limited dataset.
    - Predictions depend on the quality of the input data.
    - Model confidence does not guarantee clinical accuracy.
    - The application cannot replace professional medical diagnosis.
    """)

    st.divider()

    st.subheader("Academic Information")
    st.write("Course: ARTI303 - Programming for AI")
    st.write("Imam Abdulrahman Bin Faisal University")
    st.write("Academic Year: 2026 - Term 1")


st.divider()
st.caption(
    "For educational purposes only. "
    "Not a substitute for professional medical diagnosis."
)
