# Neural Threat Analyzer

### Hybrid Natural Language Processing and Machine Learning for Threat Detection

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)

---

**Built by [Luis Moto](https://github.com/LuisMoto) and [Nathalia Jazmín Ballesteros Luna](https://github.com/Sleepswimmings) — equal contributors.**

---

## Project Summary

A machine learning system for classifying text and payloads in a cybersecurity context. Traditional filters tend to rely on static keyword matching, which makes them vulnerable to obfuscated or context-dependent attacks.

This project addresses that limitation through a hybrid approach that combines **semantic language processing (NLP)** via neural networks with **structural feature extraction**. The pipeline classifies and detects patterns associated with:

- **Safe Content**
- **Phishing Attempts**
- **SQL Injection Attacks (SQLi)**

The system supports interactive on-demand inference, generating probability scores, and includes a metrics visualization dashboard to evaluate model performance.

---

## The Problem

The project explores a recurring technical challenge in information security:

> **How can sophisticated text-based threats — ones that evade traditional static filters — be detected using Deep Learning?**

Key questions explored during development:

- Can language models distinguish the semantics of legitimate communication from social engineering attempts?
- How does manually extracting structural markers (URLs, SQL commands, anomalous keywords) complement the vector processing of a neural model?
- How can a reproducible workflow — from data cleaning to inference — be structured for a multiclass classification problem?

---

## Tools and Technologies

| Category | Tools / Methods |
|---|---|
| Natural Language Processing (NLP) | SpaCy (`xx_ent_wiki_sm`), TextVectorization |
| Architecture and Modeling | TensorFlow, Keras, Embeddings, Dense / Attention Layers |
| Data Engineering | Python, Pandas, NumPy |
| Interface and Visualization | Streamlit, CustomTkinter |
| Version Control | Git |

---

## Directory Structure

The project is organized as follows to ensure modularity and reproducibility:

```text
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                               # Training datasets (Safe, Phishing, SQLi)
│   ├── CEAS_08.csv
│   ├── enron_spam_data.csv
│   ├── Ling.csv
│   ├── Nazario.csv
│   ├── Nigerian_Fraud.csv
│   └── SQLiV.csv
│
├── src/                                # Project source code
│   ├── app.py                          # Desktop GUI
│   ├── config.py                       # Global variables and directory paths
│   ├── dashboard.py                    # Streamlit analytics dashboard
│   ├── data_loader.py                  # Dataset ingestion, cleaning, and balancing
│   ├── features.py                     # Manual feature extraction logic
│   ├── inference_engine.py             # Prediction logic using the trained model
│   ├── preprocessing.py                # Text normalization and regex pipeline
│   └── trainer.py                      # Model architecture, compilation, and training
│
├── models/                             # Generated binary files and metrics
│   ├── metrics.json                    # Training performance history
│   ├── predictions.csv                 # Test predictions for the dashboard
│   ├── tfidf_vectorizer.pkl            # TF-IDF vectorizer (classical version)
│   ├── tfidf_vectorizer_vec.pkl        # TextVectorization config and vocabulary
│   ├── threat_classifier.keras         # Trained model weights and architecture
│   └── top_features.csv                # Feature importance analysis
│
└── docs/
    └── Case_Study_Neural_Threat_Analyzer_Luis_Moto.pdf
```
