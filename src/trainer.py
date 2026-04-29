import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.sparse import hstack
from tqdm import tqdm

from config import MODEL_PATH, VECTORIZER_PATH, RANDOM_STATE, TEST_SIZE, MAX_FEATURES
from preprocessing import security_preprocess
from data_loader import load_and_merge_data
from features import extract_features

tqdm.pandas(desc="Processing NLP (SpaCy)")

def train_model():
    print("\n--- Training model ---")

    # Load data
    df = load_and_merge_data()

    print("\n--- Distribution after balancing ---")
    print(df['Target'].value_counts())

    # Preprocessing
    print("\n--- NLP Processing ---")
    df['Clean_Text'] = df['Text'].astype(str).progress_apply(security_preprocess)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['Clean_Text'], df['Target'],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['Target']
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Extra features
    X_train_extra = np.array([extract_features(t) for t in X_train])
    X_test_extra = np.array([extract_features(t) for t in X_test])

    # Combine all features
    X_train_final = hstack([X_train_vec, X_train_extra])
    X_test_final = hstack([X_test_vec, X_test_extra])

    # Model
    model = LogisticRegression(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000
    )

    model.fit(X_train_final, y_train)

    # Cross-validation
    print("\n--- Running Cross-Validation ---")
    cv_scores = cross_val_score(model, X_train_final, y_train, cv=5)
    print(f"--- Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f}) ---")

    # Evaluation
    y_pred = model.predict(X_test_final)

    # Structured metrics 
    accuracy = accuracy_score(y_test, y_pred)
    metrics = {
        "accuracy": accuracy,
        "cv_mean_accuracy": cv_scores.mean(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # Save Models and Outputs
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("\n--- Model and Vectorizer saved ---")

    # Save metrics to JSON
    METRICS_PATH = MODEL_PATH.parent / "metrics.json"
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"--- Metrics saved to: {METRICS_PATH} ---")

    # Feature importance
    # BUG FIX: Merge TF-IDF feature names with manual feature names
    tfidf_feature_names = vectorizer.get_feature_names_out()
    manual_feature_names = ["has_url", "has_urgent", "has_verify", "has_select", "has_drop", "has_sql_comment", "has_equals"]
    all_feature_names = np.concatenate([tfidf_feature_names, manual_feature_names])

    # Class Phishing
    coefficients = model.coef_[1]
    top_features = sorted(zip(all_feature_names, coefficients), key=lambda x: x[1], reverse=True)[:20]
    
    features_df = pd.DataFrame(top_features, columns=["feature", "weight"])
    FEATURES_PATH = MODEL_PATH.parent / "top_features.csv"
    features_df.to_csv(FEATURES_PATH, index=False)
    print(f"--- Top features saved to: {FEATURES_PATH} ---")

    # Prediction dataset for Dashboard
    # Retrieve original text (using X_test which has clean text for simplicity)
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