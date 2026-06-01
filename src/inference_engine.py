import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

from config import MODEL_PATH, VECTORIZER_PATH, THREAT_CLASSES
from preprocessing import security_preprocess

KERAS_MODEL_PATH = str(MODEL_PATH).replace('.pkl', '.keras')
KERAS_VEC_PATH = str(VECTORIZER_PATH).replace('.pkl', '_vec.pkl')

try:
    print("--- Starting Deep Learning inference engine ---")
    
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    
    with open(KERAS_VEC_PATH, 'rb') as f:
        vec_data = pickle.load(f)
        
    vectorizer = TextVectorization.from_config(vec_data['config'])
    vectorizer.adapt(["dummy initialize"]) 
    vectorizer.set_weights(vec_data['weights'])
    
    print("--- Engine ready ---")
except Exception as e:
    print(f"--- ERROR loading models: {e} ---")
    model = None
    vectorizer = None

def predict_threat(raw_text):
    """
    Receives raw text and returns prediction + confidence using the Transformer.
    """
    if model is None or vectorizer is None or not raw_text.strip():
        return {
            "prediction": -1,
            "threat_name": "Error / Empty",
            "confidence": 0.0
        }

    clean_text = security_preprocess(raw_text)

    vec_text = vectorizer([clean_text])

    y_pred_probs = model.predict(vec_text, verbose=0) 
    
    prediction = np.argmax(y_pred_probs, axis=1)[0]
    confidence = np.max(y_pred_probs) * 100

    return {
        "prediction": int(prediction),
        "threat_name": THREAT_CLASSES.get(int(prediction), "Unknown"),
        "confidence": float(confidence)
    }