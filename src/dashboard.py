import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Ensure config can be imported securely
sys.path.append(str(Path(__file__).resolve().parent))
from config import MODEL_DIR

# Page Configuration
st.set_page_config(
    page_title="Malicious Email Classification",
    page_icon="shield",
    layout="wide"
)

# Style Overrides
st.markdown("""
    <style>
    /* Global Monospace terminal-style font */
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Courier Prime', monospace;
    }

    /* Force pure black background */
    .stApp {
        background-color: #000000 !important;
    }

    /* Style for metric cards (KPIs) using #09281d RGB */
    [data-testid="stMetric"] {
        background: rgba(9, 40, 29, 0.3);
        border: 1px solid rgba(9, 40, 29, 0.8);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #09281d;
    }

    /* Style for chart containers */
    [data-testid="stVerticalBlock"] > div > div > div[data-testid="stVerticalBlock"] {
        background: #050505;
        border: 1px solid #111;
        padding: 20px;
        border-radius: 10px;
    }

    /* Title customization */
    h1, h2, h3 {
        color: #ffffff !important; 
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Remove Streamlit default UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom dividers with dark green gradient */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, #09281d, transparent);
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Data Loading Logic
@st.cache_data
def load_predictions():
    return pd.read_csv(MODEL_DIR / "predictions.csv")

# Dashboard Layout
st.title("Malicious Email Classification")
st.markdown("---")

try:
    df_preds = load_predictions()

    # Section 1: Key Performance Indicators
    st.header("1. Model Performance Summary")
    
    col1, col2 = st.columns(2)
    
    # Calcular precisión directamente de las predicciones
    correct_preds = (df_preds["Real_Label"] == df_preds["Predicted_Label"]).sum()
    accuracy = correct_preds / len(df_preds)
    
    col1.metric("Global Accuracy", f"{accuracy:.4%}")
    col2.metric("Test Samples", len(df_preds))

    st.markdown("---")

    # Section 2: Confusion Matrix & Class Distribution
    st.header("2. Classification Analysis")
    
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("Confusion Matrix")
        cm = pd.crosstab(
            df_preds["Real_Label"], 
            df_preds["Predicted_Label"], 
            rownames=['Actual'], 
            colnames=['Predicted']
        )
        st.dataframe(cm.style.background_gradient(cmap='Greens'), width='stretch')
    
    with right_col:
        st.subheader("Prediction Distribution")
        dist = df_preds["Predicted_Label"].value_counts().sort_index()
        # Mapping for clarity: 0: Safe, 1: Phishing, 2: SQLi
        dist.index = ["Safe (0)", "Phishing (1)", "SQLi (2)"]
        
        st.bar_chart(dist, color="#09281d")

    st.markdown("---")

    # Section 3: Error Logs (False Positives/Negatives) 
    st.header("3. Prediction Error Analysis")
    st.markdown("Samples where the model's prediction did not match the true label.")
    
    errors = df_preds[df_preds["Real_Label"] != df_preds["Predicted_Label"]]
    
    if not errors.empty:
        st.dataframe(errors, width='stretch')
    else:
        st.success("No classification errors detected in the current test set.")

except FileNotFoundError as e:
    st.error(f"Error: Model files not found. Please run 'trainer.py' first.")
    st.info(f"Missing file: {e.filename}")

st.markdown("---")
st.caption("Malicious Email Classification | Internal Model Audit Tool")
