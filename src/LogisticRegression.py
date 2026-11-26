import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score,precision_score,recall_score

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import ALL_FEATURES, TARGET_COLUMN, build_preprocessor
import joblib
import json


def testModel(pipeline, X_test ,y_test):

    y_pred = pipeline.predict(X_test)

    return {
        "model_name": "Logistic Regression",
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4),
        "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist(),
    }



def trainmodel(data, model, metricsfile):

    df = pd.read_csv(data)

    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)
    metrics = testModel(pipeline, X_test, y_test)


    joblib.dump(pipeline, model)

    with open(metricsfile, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return pipeline, metrics


if __name__ == "__main__":
    trainmodel(
        data="heart_cleaned.csv",
        model="model.joblib",
        metricsfile="metrics.json",
    )
