import pandas as pd
import numpy as np
from sklearn.utils import resample
from config import DATA_DIR, SQLI_DATA, ENRON_DATA

def load_and_merge_data():
    print("\n--- Starting data merge ---")

    print("--- Loading safe emails (Ling + Enron Ham) ---")
    ling_df = pd.read_csv(DATA_DIR / "Ling.csv")
    ling_df = ling_df[ling_df['label'] == 0][['body']].rename(columns={'body': 'Text'})
    
    # Casual office workers
    # BYPASS: Constructing the path directly to avoid import bug
    enron_df = pd.read_csv(DATA_DIR / "enron_spam_data.csv")
    enron_df['Text'] = enron_df['Subject'].fillna('') + " " + enron_df['Message'].fillna('')
    enron_df = enron_df[enron_df['Spam/Ham'] == 'ham'][['Text']] # ONLY SAFE EMAILS
    
    safe_df = pd.concat([ling_df, enron_df], ignore_index=True)
    safe_df['Target'] = 0

    # LOAD CLASS 1: PHISHING (Nazario, CEAS, Nigerian)
    print("--- Loading Phishing attacks ---")
    
    # Nazario
    nazario = pd.read_csv(DATA_DIR / "Nazario.csv")[['body']].rename(columns={'body': 'Text'})
    # Nigerian
    nigerian = pd.read_csv(DATA_DIR / "Nigerian_Fraud.csv")[['body']].rename(columns={'body': 'Text'})
    # CEAS (Only phishing/spam)
    ceas = pd.read_csv(DATA_DIR / "CEAS_08.csv")
    ceas = ceas[ceas['label'] == 1][['body']].rename(columns={'body': 'Text'})

    phishing_df = pd.concat([nazario, nigerian, ceas], ignore_index=True)
    phishing_df['Target'] = 1

    # LOAD CLASS 2: SQLi (Technical dataset)
    print("--- Loading SQLi attacks ---")
    sqli_df = pd.read_csv(SQLI_DATA, on_bad_lines='skip')
    sqli_df['Label'] = pd.to_numeric(sqli_df['Label'], errors='coerce')
    sqli_df = sqli_df[sqli_df['Label'] == 1] # Only real attacks
    sqli_df = sqli_df[['Sentence']].rename(columns={'Sentence': 'Text'})
    sqli_df['Target'] = 2

    # UNIFICATION AND BALANCING 
    df_raw = pd.concat([safe_df, phishing_df, sqli_df], ignore_index=True).dropna()
    
    print("\n--- Applying balancing ---")
    min_size = min(df_raw['Target'].value_counts())
    
    df_balanced = pd.concat([
        resample(df_raw[df_raw['Target'] == 0], n_samples=min_size, replace=False, random_state=42),
        resample(df_raw[df_raw['Target'] == 1], n_samples=min_size, replace=False, random_state=42),
        resample(df_raw[df_raw['Target'] == 2], n_samples=min_size, replace=False, random_state=42)
    ])

    print(f"--- Dataset created. Records per class: {min_size} ---")
    return df_balanced

if __name__ == "__main__":
    df = load_and_merge_data()
    print(df['Target'].value_counts())