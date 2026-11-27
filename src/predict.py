import joblib
import pandas as pd
import logging
from pathlib import Path

try:
    from .preprocessing import ALL_FEATURES
except ImportError:
    from preprocessing import ALL_FEATURES

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
    except Exception:
        logger.error("Failed to load model")
        raise


def predict(user_data):
    try:
        model = load_model()

        input_data = pd.DataFrame(
            [user_data], columns=ALL_FEATURES
        )

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(max(probabilities)) * 100

        logger.info("Prediction completed successfully")

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2)
        }

    except Exception:
        logger.error("Prediction failed")
        raise
