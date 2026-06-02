import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# Importacion de los módulos locales del proyecto
from data_loader import load_and_merge_data
from preprocessing import security_preprocess
from features import extract_features, N_FEATURES
from config import RANDOM_STATE, MODEL_PATH, VECTORIZER_PATH

def train_model():
    print("\n--- Training model ---")

    # 1. CARGA DE DATOS
    # Consolidamos múltiples fuentes de datos (Enron, Nazario, SQLIV) para crear un corpus robusto y representativo del mundo real.
    df = load_and_merge_data()

    print("\n--- NLP Processing ---")
    tqdm.pandas()
    # 2. PREPROCESAMIENTO ORIENTADO A CIBERSEGURIDAD
    # A diferencia del NLP tradicional donde se eliminan signos y stop-words, en ciberseguridad ciertos patrones (como "1=1" o URLs extrañas) son firmas de ataques.
    # Elegimos aplicar una función custom (security_preprocess) para preservar y abstraer estos tokens críticos (ej. 'sql_tautology', 'url_token') evitando que se pierda la señal maliciosa.
    df['Clean_Text'] = df['Text'].astype(str).progress_apply(security_preprocess)

    # 3. PARTICIÓN DE DATOS (DATA SPLIT)
    # Elegimos una división 70/15/15 (Train/Val/Test).
    # Es crucial el uso de 'stratify=df['Target']'. Los datasets de ciberseguridad están naturalmente desbalanceados (muchos correos seguros, pocos ataques). Si hiciéramos un split aleatorio, podríamos dejar el set de validación sin ejemplos de SQLi.
    # Stratify garantiza que la distribución de probabilidad de las clases se mantenga constante.
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['Clean_Text'], df['Target'],
        test_size=0.30, random_state=RANDOM_STATE, stratify=df['Target']
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    print("\n--- Extracting manual features ---")
    # 4. EXTRACCIÓN DE CARACTERÍSTICAS ESTRUCTURALES (RAMA 1)
    # Los modelos de lenguaje puro a veces fallan al capturar reglas lógicas duras. Decidimos extraer heurísticas manuales (como longitud del texto, conteo de caracteres especiales) para darle al modelo un vector numérico denso que complemente el análisis semántico.
    X_train_feats = np.array([extract_features(t) for t in X_train], dtype=np.float32)
    X_val_feats   = np.array([extract_features(t) for t in X_val], dtype=np.float32)
    X_test_feats  = np.array([extract_features(t) for t in X_test], dtype=np.float32)
    
    print(f"Manual feature matrix shape: {X_train_feats.shape}  ({N_FEATURES} features)")

    print("\n--- Text Vectorization ---")
    # 5. VECTORIZACIÓN DE TEXTO (RAMA 2)
    max_tokens = 8000       # Creamos un máximo de tokens
    sequence_length = 256   # Truncamos/hacemos padding a 256 para estandarizar los tensores de entrada.

    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length
    )
    
    # Adaptamos el vocabulario EXCLUSIVAMENTE con X_train. Si usáramos todo el dataset, cometeríamos 'Data Leakage' (fuga de datos), ya que  el modelo conocería palabras del set de prueba antes de ser evaluado.
    vectorizer.adapt(X_train)

    X_train_vec = vectorizer(X_train)
    X_val_vec   = vectorizer(X_val)
    X_test_vec  = vectorizer(X_test)

    print("\n--- Building Dual-Input Transformer Encoder ---")
    
    # ARQUITECTURA DEL MODELO: ENFOQUE HÍBRIDO (DUAL-STREAM)
    # Elegimos una arquitectura de dos entradas para que el modelo aprenda tanto del contexto profundo del texto como de los indicadores matemáticos duros.

    # --- STREAM SEMÁNTICO (TRANSFORMER) ---
    text_input = Input(shape=(sequence_length,), name="text_input")
    # Mapeamos índices enteros a vectores densos en un espacio latente de 64 dimensiones.
    embedding = layers.Embedding(input_dim=max_tokens, output_dim=64)(text_input)

    # Elegimos MultiHeadAttention en lugar de LSTMs porque la atención calcula matemáticamente la correlación entre TODAS las palabras simultáneamente, capturando dependencias a largo plazo (ej. "urgente" al inicio y un "enlace" al final).
    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=32)(embedding, embedding)
    attn_output = layers.Dropout(0.1)(attn_output) # Regularización para evitar over-fitting
    
    # Conexión Residual (Add): Evita el problema del desvanecimiento del gradiente.
    out1 = layers.Add()([embedding, attn_output])
    # LayerNormalization: Estabiliza el aprendizaje manteniendo la media en 0 y varianza en 1.
    out1 = layers.LayerNormalization(epsilon=1e-6)(out1)
    
    ffn_output = layers.Dense(64, activation="relu")(out1)
    ffn_output = layers.Dense(64)(ffn_output)
    ffn_output = layers.Dropout(0.1)(ffn_output)
    
    out2 = layers.Add()([out1, ffn_output])
    seq_out = layers.LayerNormalization(epsilon=1e-6)(out2)

    # GlobalMaxPooling1D: En vez de aplanar la matriz, extraemos la señal más fuerte de toda la secuencia. Si hay un solo token que grita "SQL Injection", esto lo priorizará.
    text_vec = layers.GlobalMaxPooling1D()(seq_out)
    text_dense = layers.Dense(16, activation="relu")(text_vec)

    # --- STREAM ESTRUCTURAL (MANUAL FEATURES) ---
    feat_input = Input(shape=(N_FEATURES,), name="feat_input")
    feat_dense = layers.Dense(32, activation="relu")(feat_input)
    # Aplicamos BatchNormalization porque las características manuales tienen escalas numéricas muy distintas. Esto normaliza los lotes y ayuda al gradiente a converger.
    feat_dense = layers.BatchNormalization()(feat_dense)
    feat_out = layers.Dense(16, activation="relu")(feat_dense)

    # --- FUSIÓN (CONCATENACIÓN) ---
    # Unimos la representación matemática del texto con la lógica estructural.
    concat = layers.Concatenate()([text_dense, feat_out])
    concat = layers.Dropout(0.2)(concat)
    concat = layers.Dense(64, activation="relu")(concat)
    concat = layers.Dropout(0.2)(concat)

    # Capa de salida: Softmax distribuye la probabilidad matemática entre las 3 clases.
    outputs = layers.Dense(3, activation="softmax")(concat)

    model = Model(inputs=[text_input, feat_input], outputs=outputs)
    
    # Compilamos usando Adam para una convergencia más rápida y crossentropy para multi-clase.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )
    
    model.summary()

    # 6. MANEJO MATEMÁTICO DEL DESBALANCE DE CLASES
    # En lugar de borrar datos útiles de la clase mayoritaria (undersampling), calculamos pesos inversamente proporcionales a la frecuencia de la clase.
    # Si la red clasifica mal un ataque (minoría), el error penaliza mucho más fuerte en la función de costo.
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    weights_dict = {cls: weight for cls, weight in zip(classes, weights)}
    print(f"\nClass weights: {weights_dict}")

    keras_model_path = str(MODEL_PATH).replace('.pkl', '.keras')
    
    # 7. ESTRATEGIA DE ENTRENAMIENTO Y CALLBACKS
    callbacks = [
        # EarlyStopping: Si el 'val_loss' sube durante 3 épocas, el modelo está memorizando (overfitting). Detenemos el entrenamiento y restauramos los pesos del mejor punto global.
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        # ReduceLROnPlateau: Si el aprendizaje se estanca en una meseta local, reducimos dinámicamente  el 'learning rate' para dar pasos más pequeños y encontrar el mínimo global de la función de pérdida.
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5),
        ModelCheckpoint(keras_model_path, save_best_only=True, monitor='val_loss')
    ]

    print("\n--- Training ---")
    history = model.fit(
        [X_train_vec, X_train_feats], y_train,
        validation_data=([X_val_vec, X_val_feats], y_val),
        epochs=25, batch_size=32,
        callbacks=callbacks, class_weight=weights_dict
    )

    print("\n--- Evaluating on Test Set ---")
    loss, accuracy = model.evaluate([X_test_vec, X_test_feats], y_test)
    print(f"Test Loss: {loss:.4f}  |  Test Accuracy: {accuracy:.4f}")

    y_pred_probs = model.predict([X_test_vec, X_test_feats])
    y_pred       = np.argmax(y_pred_probs, axis=1)

    # 8. EVALUACIÓN DE DESEMPEÑO
    # En dominios desbalanceados la "Accuracy" global miente. Por eso imprimimos la matriz de confusión y el reporte detallado (Precision, Recall, F1-Score), para asegurar que estamos detectando los falsos negativos en la clase Phishing y SQLi.
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    print("\n--- Classification Report ---")
    print(classification_report(
        y_test, y_pred,
        target_names=["Safe (0)", "Phishing (1)", "SQLi (2)"]
    ))

    # Guardamos el vectorizador para mantener la coherencia del vocabulario en producción.
    vec_path = str(VECTORIZER_PATH).replace('.pkl', '_vec.pkl')
    with open(vec_path, 'wb') as f:
        pickle.dump({
            'config':     vectorizer.get_config(),
            'vocabulary': vectorizer.get_vocabulary()
        }, f)

    print(f"\n--- Model saved  → {keras_model_path}")
    print(f"--- Vectorizer saved → {vec_path}")

    # Guardamos las inferencias para posible análisis iterativo o consumo en un dashboard.
    results_df = pd.DataFrame({
        'Real_Label': y_test,
        'Predicted_Label': y_pred
    })
    results_df.to_csv(str(MODEL_PATH).replace('threat_classifier.pkl', 'predictions.csv'), index=False)

if __name__ == "__main__":
    train_model()
