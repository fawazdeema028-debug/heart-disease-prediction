# Heart Disease Prediction

A machine learning application that predicts the likelihood of heart disease based on clinical features using the Kaggle Heart Failure Prediction dataset, with an interactive Streamlit interface.

## Project Structure
- data/: Contains the dataset (`heart.csv`).
- src/: Scripts for data preprocessing, model training, and prediction.
- app/: Streamlit web interface.
- tests/: Unit and integration tests with sample inputs and expected outputs.
- requirements.txt: Python dependencies required to run the project.

## Dataset
- Source: [Heart Failure Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- Target Variable: HeartDisease (1 = Heart Disease, 0 = Normal)

## Setup & Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd heart-failure-prediction
   Install required libraries:
   pip install -r requirements.txt
   Run the Streamlit app:
   streamlit run app/app.py
   Run tests:
   pytest
   Team Responsibilities
 Member 1: Data Processing & Preprocessing
 Member 2: Machine Learning Algorithms
 Member 3: Streamlit Frontend
 Member 4: Backend & Integration
 Member 5: Testing & Reproducibility
