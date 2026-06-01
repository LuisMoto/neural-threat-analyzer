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
from features import extract_features, N_FEATURES
from config import RANDOM_STATE, MODEL_PATH, VECTORIZER_PATH


def train_model():
    print("\n--- Training model ---")

    df = load_and_merge_data()

    print("\n--- NLP Processing ---")
    tqdm.pandas()
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

    X_train_np = X_train.fillna("").astype(str).to_numpy()
    X_val_np   = X_val.fillna("").astype(str).to_numpy()
    X_test_np  = X_test.fillna("").astype(str).to_numpy()

    print("\n--- Extracting manual features ---")
    X_train_feats = np.array([extract_features(t) for t in X_train_np], dtype=np.float32)
    X_val_feats   = np.array([extract_features(t) for t in X_val_np],   dtype=np.float32)
    X_test_feats  = np.array([extract_features(t) for t in X_test_np],  dtype=np.float32)

    print(f"Manual feature matrix shape: {X_train_feats.shape}  ({N_FEATURES} features)")

    print("\n--- Text Vectorization ---")
    VOCAB_SIZE          = 10_000
    MAX_SEQUENCE_LENGTH = 256

    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH
    )
    vectorizer.adapt(X_train_np)

    X_train_vec = vectorizer(X_train_np)
    X_val_vec   = vectorizer(X_val_np)
    X_test_vec  = vectorizer(X_test_np)

    print("\n--- Building Dual-Input Transformer Encoder ---")

    embed_dim  = 64   
    num_heads  = 4    
    ff_dim     = 64


    text_input      = Input(shape=(MAX_SEQUENCE_LENGTH,), name="text_input")
    x               = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=embed_dim)(text_input)


    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn_out = layers.Dropout(0.1)(attn_out)
    out1     = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

    ffn_out  = layers.Dense(ff_dim, activation="relu")(out1)
    ffn_out  = layers.Dense(embed_dim)(ffn_out)
    ffn_out  = layers.Dropout(0.1)(ffn_out)
    seq_out  = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_out)

    text_vec = layers.GlobalAveragePooling1D()(seq_out)  


    feat_input = Input(shape=(N_FEATURES,), name="feat_input")
    feat_dense = layers.Dense(32, activation="relu")(feat_input)
    feat_dense = layers.BatchNormalization()(feat_dense)
    feat_dense = layers.Dense(16, activation="relu")(feat_dense)

    merged  = layers.Concatenate()([text_vec, feat_dense])
    merged  = layers.Dropout(0.3)(merged)
    merged  = layers.Dense(64, activation="relu")(merged)
    merged  = layers.Dropout(0.2)(merged)
    outputs = layers.Dense(3, activation="softmax")(merged)

    model = Model(inputs=[text_input, feat_input], outputs=outputs)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )

    keras_model_path = str(MODEL_PATH).replace('.pkl', '.keras')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
        ModelCheckpoint(filepath=keras_model_path, monitor='val_loss', save_best_only=True),
    ]

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    weights_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {weights_dict}")

    print("\n--- Training ---")
    history = model.fit(
        [X_train_vec, X_train_feats], y_train,
        validation_data=([X_val_vec, X_val_feats], y_val),
        epochs=25,
        batch_size=32,
        callbacks=callbacks,
        class_weight=weights_dict
    )

    print("\n--- Evaluating on Test Set ---")
    loss, accuracy = model.evaluate([X_test_vec, X_test_feats], y_test)
    print(f"Test Loss: {loss:.4f}  |  Test Accuracy: {accuracy:.4f}")

    y_pred_probs = model.predict([X_test_vec, X_test_feats])
    y_pred       = np.argmax(y_pred_probs, axis=1)

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    print("\n--- Classification Report ---")
    print(classification_report(
        y_test, y_pred,
        target_names=["Safe (0)", "Phishing (1)", "SQLi (2)"]
    ))


    vectorizer_data = {
        'config':     vectorizer.get_config(),
        'vocabulary': vectorizer.get_vocabulary()
    }
    vec_path = str(VECTORIZER_PATH).replace('.pkl', '_vec.pkl')
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer_data, f)

    print(f"\n--- Model saved  → {keras_model_path}")
    print(f"--- Vectorizer saved → {vec_path}")

    results_df = pd.DataFrame({
        "Clean_Text":      X_test,
        "Real_Label":      y_test.values,
        "Predicted_Label": y_pred
    })
    PREDS_PATH = MODEL_PATH.parent / "predictions.csv"
    results_df.to_csv(PREDS_PATH, index=False)
    print(f"--- Predictions saved → {PREDS_PATH}")


if __name__ == "__main__":
    train_model()