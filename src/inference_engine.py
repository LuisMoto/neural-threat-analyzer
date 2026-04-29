import joblib
import numpy as np
from scipy.sparse import hstack

from config import MODEL_PATH, VECTORIZER_PATH, THREAT_CLASSES
from preprocessing import security_preprocess
from features import extract_features

# Load models
try:
    print("--- Starting inference engine ---")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("--- Engine ready ---")
except FileNotFoundError:
    print("--- ERROR: Models not found. Run train_model first ---")
    model = None
    vectorizer = None


def predict_threat(raw_text):
    """
    Receives raw text and returns prediction + confidence.
    """
    # sklearn objects overload __bool__ and can raise ValueError.
    if model is None or vectorizer is None or not raw_text.strip():
        return {
            "prediction": -1,
            "threat_name": "Error / Empty",
            "confidence": 0.0
        }

    # 1. Preprocessing
    clean_text = security_preprocess(raw_text)

    # 2. TF-IDF
    vec_text = vectorizer.transform([clean_text])

    # 3. Manual features
    extra_features = np.array([extract_features(clean_text)])

    # 4. Combine
    final_input = hstack([vec_text, extra_features])

    # 5. Prediction
    prediction = model.predict(final_input)[0]
    confidence = model.predict_proba(final_input).max() * 100

    return {
        "prediction": int(prediction),
        "threat_name": THREAT_CLASSES.get(prediction, "Unknown"),
        "confidence": float(confidence)
    }