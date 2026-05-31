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

from data_loader import load_and_merge_data
from preprocessing import security_preprocess
from config import RANDOM_STATE, MODEL_PATH, VECTORIZER_PATH

def train_model():
    print("\n--- Training model ---")

    # Load data
    df = load_and_merge_data()

    print("\n--- Distribution after balancing ---")
    print(df['Target'].value_counts())

    # Preprocessing
    print("\n--- NLP Processing ---")
    tqdm.pandas()  # <--- AQUÍ ESTÁ LA INICIALIZACIÓN FALTANTE
    df['Clean_Text'] = df['Text'].astype(str).progress_apply(security_preprocess)

    X_train, X_temp, y_train, y_temp = train_test_split(
        df['Clean_Text'], df['Target'],
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df['Target']
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    # 2. Vectorización para Deep Learning
    print("\n--- Text Vectorization ---")
    VOCAB_SIZE = 10000
    MAX_SEQUENCE_LENGTH = 256

    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH
    )

    # Adaptar el vocabulario solo con los datos de entrenamiento
    vectorizer.adapt(X_train)

    # Transformar los textos en tensores de enteros
    X_train_vec = vectorizer(X_train)
    X_val_vec = vectorizer(X_val)
    X_test_vec = vectorizer(X_test)

    # 3. Construcción del Transformer Encoder
    print("\n--- Building Transformer Encoder ---")
    embed_dim = 32  # Tamaño del vector para cada token
    num_heads = 2   # Número de cabezas de atención
    ff_dim = 32     # Tamaño de la red feed-forward oculta

    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    
    # Capa de Embedding
    embedding_layer = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=embed_dim)(inputs)

    # Bloque de Atención
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(embedding_layer, embedding_layer)
    attention_output = layers.Dropout(0.1)(attention_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(embedding_layer + attention_output)

    # Red Feed-Forward
    ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    ffn_output = layers.Dropout(0.1)(ffn_output)
    sequence_output = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    # Capas de salida
    x = layers.GlobalAveragePooling1D()(sequence_output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(3, activation="softmax")(x) 

    model = Model(inputs=inputs, outputs=outputs)

    # 4. Compilación y Configuración de Callbacks
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )

    keras_model_path = str(MODEL_PATH).replace('.pkl', '.keras')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
        ModelCheckpoint(filepath=keras_model_path, monitor='val_loss', save_best_only=True)
    ]

    # 5. Entrenamiento
    print("\n--- Training Deep Learning Model ---")
    
    # Calcular pesos de clase
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    history = model.fit(
        X_train_vec, y_train,
        validation_data=(X_val_vec, y_val),
        epochs=20, 
        batch_size=32,
        callbacks=callbacks,
        class_weight=weights_dict
    )

    # 6. Evaluación
    print("\n--- Evaluating Model on Test Set ---")
    loss, accuracy = model.evaluate(X_test_vec, y_test)
    
    # Predicciones para reporte
    y_pred_probs = model.predict(X_test_vec)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # 7. Guardar modelo y vectorizador
    vectorizer_data = {
        'config': vectorizer.get_config(),
        'weights': vectorizer.get_weights()
    }
    
    with open(str(VECTORIZER_PATH).replace('.pkl', '_vec.pkl'), 'wb') as f:
        pickle.dump(vectorizer_data, f)

    print(f"\n--- Model saved to: {keras_model_path} ---")
    
    # Dataset de predicciones para Dashboard
    results_df = pd.DataFrame({
        "Clean_Text": X_test,
        "Real_Label": y_test,
        "Predicted_Label": y_pred
    })
    PREDS_PATH = MODEL_PATH.parent / "predictions.csv"
    results_df.to_csv(PREDS_PATH, index=False)
    print(f"--- Test predictions saved to: {PREDS_PATH} ---")

if __name__ == "__main__":
    train_model()