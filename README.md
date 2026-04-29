# Neural Threat Analyzer  

### Hybrid NLP & Structural Machine Learning for Threat Detection

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Deployed-success?style=for-the-badge)

---

## Project Documentation
For a deep dive into the business context, methodology, and strategic roadmap, please refer to the:
* **[Executive & Technical Case Study (PDF)](./docs/Case_Study_Neural_Threat_Analyzer_Luis_Moto.pdf)**

## Executive Summary

**Neural Threat Analyzer** is a machine learning system designed to classify text-based payloads for cybersecurity applications. Traditional filters often rely on static keyword detection, making them vulnerable to obfuscated or context-dependent attacks.

This project addresses that limitation by combining **semantic language processing (NLP)** with **engineered structural indicators**, creating a hybrid pipeline capable of detecting patterns associated with:

- **Safe Content**
- **Phishing Attempts**
- **SQL Injection Attacks (SQLi)**

The system performs **interactive, on-demand classification**, generating probabilistic confidence scores alongside interpretable feature analysis to support analyst decision-making.

---

## Project Documentation
For a deep dive into the business context, methodology, and strategic roadmap, please refer to the:
* **[Executive & Technical Case Study (PDF)](./docs/Case_Study_Neural_Threat_Analyzer_Luis_Moto.pdf)**

---

## Business & Security Objective

The project explores a critical challenge in modern cybersecurity:

> **How can organizations detect sophisticated text-based threats that evade traditional static filters?**

Key questions addressed:

- Can linguistic patterns differentiate legitimate communication from social engineering attempts?
- How can structural markers (URLs, SQL syntax, encoded payloads) complement semantic understanding?
- How can explainable outputs improve analyst trust and operational efficiency?

---

## Tech Stack

| Category | Tools / Methods |
|---|---|
| Natural Language Processing | SpaCy (`en_core_web_sm`) |
| Machine Learning Pipeline | Scikit-learn, Logistic Regression, TF-IDF |
| Data Engineering | Python, Pandas, NumPy |
| Interface & Explainability | Streamlit, CustomTkinter |
| Version Control & Deployment | Git, Streamlit Cloud |

---

## Project Architecture

```bash
├── data/
│   ├── CEAS_08.csv             # Phishing dataset
│   ├── enron_spam_data.csv     # Safe content dataset
│   ├── SQLiV.csv               # SQL injection dataset
│   └── ... (additional sources)
│
├── src/
│   ├── config.py               # Path management & global variables
│   ├── data_loader.py          # Data ingestion & balancing logic
│   ├── preprocessing.py        # NLP normalization pipeline
│   ├── features.py             # Structural feature extraction
│   ├── trainer.py              # Model training & evaluation
│   ├── inference_engine.py     # Live inference logic
│   ├── app.py                  # Desktop GUI
│   └── dashboard.py            # Streamlit analytics interface
│
├── models/
│   ├── threat_classifier.pkl   # Trained ML model
│   ├── tfidf_vectorizer.pkl    # Vocabulary mapping
│   └── metrics.json            # Performance records
│
├── requirements.txt
└── .gitignore
