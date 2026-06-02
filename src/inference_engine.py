import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

# Importaciones de configuración y preprocesamiento de nuestro pipeline
from config import MODEL_PATH, VECTORIZER_PATH, THREAT_CLASSES
from preprocessing import security_preprocess
from features import extract_features, N_FEATURES

# Adaptamos las rutas para cargar el formato nativo .keras, que es más eficiente para el grafo computacional que el antiguo formato h5.
KERAS_MODEL_PATH = str(MODEL_PATH).replace('.pkl', '.keras')
KERAS_VEC_PATH   = str(VECTORIZER_PATH).replace('.pkl', '_vec.pkl')


# INICIALIZACIÓN GLOBAL (SINGLETON / EAGER LOADING)

# El objetivo principal del proyecto es lograr un tiempo medio de detección (MTTD) reducido
# Para lograr esa identificación casi en tiempo real, cargamos el modelo pesado y el vectorizador en la memoria global una sola vez al arrancar el script, en lugar de 
# hacerlo dentro de la función predict_threat. Esto elimina cuellos de botella de I/O por cada petición.
try:
    print("--- Starting inference engine ---")
    
    # Cargamos el grafo computacional completo y los pesos entrenados del Transformer
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)

    # Reconstrucción del Vectorizador: Para evitar el "Train-Serve Skew" (sesgo donde el modelo en producción se comporta distinto al de entrenamiento), reconstruimos la capa TextVectorization exactamente con la misma configuración y el vocabulario adaptado previamente.
    with open(KERAS_VEC_PATH, 'rb') as f:
        vec_data = pickle.load(f)

    vectorizer = TextVectorization.from_config(vec_data['config'])
    vectorizer.set_vocabulary(vec_data['vocabulary'])

    print(f"--- Engine ready (inputs: {len(model.inputs)}) ---")

except Exception as e:
    # Manejo de excepciones robusto. Si este motor se va a desplegar como microservicio 
    # (ej. FastAPI) o en un Web Application Firewall (WAF) no podemos permitir que un error de lectura de archivo tire todo el servidor.
    print(f"--- ERROR loading models: {e} ---")
    model      = None
    vectorizer = None


def predict_threat(raw_text: str) -> dict:
    """
    Toma una cadena de texto cruda, aplica el Dual-Stream Feature Pipeline,
    y devuelve la predicción matemática junto con las probabilidades de Softmax.
    """
    
    # 1. VALIDACIONES DE SEGURIDAD Y RENDIMIENTO
    if model is None or vectorizer is None:
        return {"prediction": -1, "threat_name": "Engine not loaded", "confidence": 0.0}

    # Descartamos entradas vacías con una complejidad temporal O(1). Esto ahorra ciclos valiosos de CPU evitando cálculos matriciales innecesarios.
    if not raw_text.strip():
        return {"prediction": -1, "threat_name": "Empty input", "confidence": 0.0}


    # 2. TRANSFORMACIÓN DE DATOS (PIPELINE DE INFERENCIA)

    # Es vital aplicar exactamente las mismas reglas de preprocesamiento para garantizar que la sensibilidad del modelo no se diluya ante datos reales
    
    # A. Normalización y abstracción de tokens (e.g. preservación de keywords maliciosas)
    clean_text   = security_preprocess(raw_text)
    
    # B. Rama Semántica: Mapeo del texto al espacio latente (Enteros 1D)
    vec_text     = vectorizer([clean_text])
    
    # C. Rama Estructural: Extracción al vuelo de las características manuales densas
    manual_feats = np.array([extract_features(clean_text)], dtype=np.float32)

    # 3. PROPAGACIÓN HACIA ADELANTE (FORWARD PASS)

    # Justificación: verbose=0 es obligatorio en inferencia de producción para no saturar los logs del sistema operativo ni ralentizar el tiempo de respuesta.
    # Se alimentan ambas ramas de forma simultánea a las dos capas Input de Keras.
    y_pred_probs = model.predict([vec_text, manual_feats], verbose=0)


    # 4. PARSEO DE RESULTADOS MATEMÁTICOS

    # np.argmax extrae el índice de clase (0, 1, o 2) con la probabilidad más alta.
    prediction = int(np.argmax(y_pred_probs, axis=1)[0])
    
    # Extraemos el valor máximo de probabilidad y lo pasamos a porcentaje.
    confidence = float(np.max(y_pred_probs) * 100)

    # Formato de retorno:
    # Devolvemos un diccionario estructurado para facilitar la integración con el dashboard de monitoreo operativo (Streamlit). Además, incluir el desglose exacto de las probabilidades ('probabilities') sentará las bases matemáticas para analizar la matriz de confusión  y para la futura fase de Explainable AI, facilitando las investigaciones forenses
    return {
        "prediction":  prediction,
        "threat_name": THREAT_CLASSES.get(prediction, "Unknown"),
        "confidence":  confidence,
        "probabilities": {
            THREAT_CLASSES.get(i, str(i)): float(p * 100)
            for i, p in enumerate(y_pred_probs[0])
        }
    }
