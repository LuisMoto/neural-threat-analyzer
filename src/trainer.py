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
        test_size=0.30, random_state=RANDOM_STATE, stratify=df['Target']
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    print("\n--- Extracting manual features ---")
    X_train_feats = np.array([extract_features(t) for t in X_train], dtype=np.float32)
    X_val_feats   = np.array([extract_features(t) for t in X_val], dtype=np.float32)
    X_test_feats  = np.array([extract_features(t) for t in X_test], dtype=np.float32)
    
    print(f"Manual feature matrix shape: {X_train_feats.shape}  ({N_FEATURES} features)")

  
    print("\n--- Text Vectorization ---")
    max_tokens = 8000
    sequence_length = 256

    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length
    )
    vectorizer.adapt(X_train)

    X_train_vec = vectorizer(X_train)
    X_val_vec   = vectorizer(X_val)
    X_test_vec  = vectorizer(X_test)

    print("\n--- Building Dual-Input Transformer Encoder ---")
  
    text_input = Input(shape=(sequence_length,), name="text_input")
    embedding = layers.Embedding(input_dim=max_tokens, output_dim=64)(text_input)

    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=32)(embedding, embedding)
    attn_output = layers.Dropout(0.1)(attn_output)
    out1 = layers.Add()([embedding, attn_output])
    out1 = layers.LayerNormalization(epsilon=1e-6)(out1)
    
    ffn_output = layers.Dense(64, activation="relu")(out1)
    ffn_output = layers.Dense(64)(ffn_output)
    ffn_output = layers.Dropout(0.1)(ffn_output)
    
    out2 = layers.Add()([out1, ffn_output])
    seq_out = layers.LayerNormalization(epsilon=1e-6)(out2)

    text_vec = layers.GlobalMaxPooling1D()(seq_out)
    text_dense = layers.Dense(16, activation="relu")(text_vec)

    feat_input = Input(shape=(N_FEATURES,), name="feat_input")
    feat_dense = layers.Dense(32, activation="relu")(feat_input)
    feat_dense = layers.BatchNormalization()(feat_dense)
    feat_out = layers.Dense(16, activation="relu")(feat_dense)

    concat = layers.Concatenate()([text_dense, feat_out])
    concat = layers.Dropout(0.2)(concat)
    concat = layers.Dense(64, activation="relu")(concat)
    concat = layers.Dropout(0.2)(concat)

    outputs = layers.Dense(3, activation="softmax")(concat)

    model = Model(inputs=[text_input, feat_input], outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )
    
    model.summary()

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    weights_dict = {cls: weight for cls, weight in zip(classes, weights)}
    print(f"\nClass weights: {weights_dict}")

    keras_model_path = str(MODEL_PATH).replace('.pkl', '.keras')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
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

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    print("\n--- Classification Report ---")
    print(classification_report(
        y_test, y_pred,
        target_names=["Safe (0)", "Phishing (1)", "SQLi (2)"]
    ))


    vec_path = str(VECTORIZER_PATH).replace('.pkl', '_vec.pkl')
    with open(vec_path, 'wb') as f:
        pickle.dump({
            'config':     vectorizer.get_config(),
            'vocabulary': vectorizer.get_vocabulary()
        }, f)

    print(f"\n--- Model saved  → {keras_model_path}")
    print(f"--- Vectorizer saved → {vec_path}")


    results_df = pd.DataFrame({
        'Real_Label': y_test,
        'Predicted_Label': y_pred
    })
    results_df.to_csv(str(MODEL_PATH).replace('threat_classifier.pkl', 'predictions.csv'), index=False)

if __name__ == "__main__":
    train_model()