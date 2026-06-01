import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

from config import MODEL_PATH, VECTORIZER_PATH, THREAT_CLASSES
from preprocessing import security_preprocess
from features import extract_features, N_FEATURES

KERAS_MODEL_PATH = str(MODEL_PATH).replace('.pkl', '.keras')
KERAS_VEC_PATH   = str(VECTORIZER_PATH).replace('.pkl', '_vec.pkl')

try:
    print("--- Starting inference engine ---")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)

    with open(KERAS_VEC_PATH, 'rb') as f:
        vec_data = pickle.load(f)

    vectorizer = TextVectorization.from_config(vec_data['config'])
    vectorizer.set_vocabulary(vec_data['vocabulary'])

    print(f"--- Engine ready (inputs: {len(model.inputs)}) ---")

except Exception as e:
    print(f"--- ERROR loading models: {e} ---")
    model      = None
    vectorizer = None


def predict_threat(raw_text: str) -> dict:
    if model is None or vectorizer is None:
        return {"prediction": -1, "threat_name": "Engine not loaded", "confidence": 0.0}

    if not raw_text.strip():
        return {"prediction": -1, "threat_name": "Empty input", "confidence": 0.0}


    clean_text   = security_preprocess(raw_text)
    vec_text     = vectorizer([clean_text])
    manual_feats = np.array([extract_features(clean_text)], dtype=np.float32)

    y_pred_probs = model.predict([vec_text, manual_feats], verbose=0)

    prediction = int(np.argmax(y_pred_probs, axis=1)[0])
    confidence = float(np.max(y_pred_probs) * 100)

    return {
        "prediction":  prediction,
        "threat_name": THREAT_CLASSES.get(prediction, "Unknown"),
        "confidence":  confidence,
        "probabilities": {
            THREAT_CLASSES.get(i, str(i)): float(p * 100)
            for i, p in enumerate(y_pred_probs[0])
        }
    }