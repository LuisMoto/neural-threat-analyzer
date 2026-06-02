# Neural Threat Analyzer  

### Procesamiento de Lenguaje Natural y Aprendizaje Automático Híbrido para la Detección de Amenazas

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)

---

## Resumen del Proyecto

**Neural Threat Analyzer** es un sistema de aprendizaje automático diseñado para clasificar textos y cargas útiles (payloads) en el contexto de la ciberseguridad. Los filtros tradicionales suelen depender de la detección estática de palabras clave, lo que los hace vulnerables a ataques ofuscados o dependientes del contexto.

Este proyecto aborda esa limitación implementando un enfoque híbrido que combina el **procesamiento semántico del lenguaje (NLP)** mediante redes neuronales con la **extracción de características estructurales**. Este *pipeline* es capaz de clasificar y detectar patrones asociados con:

- **Contenido Seguro**
- **Intentos de Phishing**
- **Ataques de Inyección SQL (SQLi)**

El sistema permite realizar inferencia interactiva bajo demanda, generando puntajes de probabilidad e integrando un panel de visualización de métricas para evaluar el rendimiento del modelo.

---

## Documentación
Para una revisión detallada de la metodología, el preprocesamiento de datos y el diseño de la arquitectura de la red, consultar el siguiente documento:
* **[Caso de Estudio y Documentación Técnica (PDF)](./docs/Case_Study_Neural_Threat_Analyzer_Luis_Moto.pdf)**

---

## Planteamiento del Problema

El proyecto explora un desafío técnico recurrente en la seguridad informática:

> **¿Cómo se pueden detectar amenazas sofisticadas basadas en texto que evaden los filtros estáticos tradicionales mediante el uso de Deep Learning?**

Preguntas clave abordadas durante el desarrollo:

- ¿Pueden los modelos lingüísticos diferenciar la semántica de una comunicación legítima frente a la de un intento de ingeniería social?
- ¿De qué manera la extracción manual de marcadores estructurales (URLs, comandos SQL, palabras clave anómalas) complementa el procesamiento vectorial de un modelo neuronal?
- ¿Cómo se puede estructurar un flujo de trabajo reproducible (desde la limpieza de datos hasta la inferencia) para un problema de clasificación multiclase?

---

## Herramientas y Tecnologías

| Categoría | Herramientas / Métodos |
|---|---|
| Procesamiento de Lenguaje (NLP) | SpaCy (`xx_ent_wiki_sm`), TextVectorization |
| Arquitectura y Modelado | TensorFlow, Keras, Embeddings, Capas Densas / Atención |
| Ingeniería de Datos | Python, Pandas, NumPy |
| Interfaz y Visualización | Streamlit, CustomTkinter |
| Control de Versiones | Git |

---

## Estructura del Directorio

El proyecto está organizado de la siguiente manera para asegurar la modularidad y reproducibilidad del código:

```text
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                               # Datasets de entrenamiento (Safe, Phishing, SQLi)
│   ├── CEAS_08.csv
│   ├── enron_spam_data.csv
│   ├── Ling.csv
│   ├── Nazario.csv
│   ├── Nigerian_Fraud.csv
│   └── SQLiV.csv
│
├── src/                                # Código fuente del proyecto
│   ├── app.py                          # Interfaz gráfica de escritorio (GUI)
│   ├── config.py                       # Variables globales y rutas de directorios
│   ├── dashboard.py                    # Aplicación web analítica en Streamlit
│   ├── data_loader.py                  # Ingesta, limpieza y balanceo de datasets
│   ├── features.py                     # Lógica de extracción de características manuales
│   ├── inference_engine.py             # Lógica de predicción con el modelo entrenado
│   ├── preprocessing.py                # Pipeline de normalización de texto y regex
│   └── trainer.py                      # Arquitectura del modelo, compilación y entrenamiento
│
├── models/                             # Archivos binarios y métricas generadas
│   ├── metrics.json                    # Historial de rendimiento del entrenamiento
│   ├── predictions.csv                 # Predicciones de prueba para el dashboard
│   ├── tfidf_vectorizer.pkl            # Vectorizador TF-IDF (versión clásica)
│   ├── tfidf_vectorizer_vec.pkl        # Configuración y vocabulario de TextVectorization
│   ├── threat_classifier.keras         # Pesos y arquitectura del modelo entrenado
│   └── top_features.csv                # Análisis de importancia de variables
│
└── docs/
    └── Case_Study_Neural_Threat_Analyzer_Luis_Moto.pdf
└── .gitignore
