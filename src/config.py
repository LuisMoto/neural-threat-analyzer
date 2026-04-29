# src_config

from pathlib import Path

# Path Management
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset Paths
DATA_DIR = BASE_DIR / "data"
ENRON_DATA = DATA_DIR / "enron_spam_data.csv"
SQLI_DATA = DATA_DIR / "SQLiV.csv"

# Model Paths
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "threat_classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


# Threat Mapping
THREAT_CLASSES = {
    0: "Safe Content",
    1: "Phishing Attempt",
    2: "SQL Injection Attack"
}


# Config
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 8000 # Vocabulary limit for TF-IDF