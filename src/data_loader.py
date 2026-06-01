import pandas as pd
import numpy as np
from sklearn.utils import resample
from config import DATA_DIR, SQLI_DATA, ENRON_DATA


def load_and_merge_data():
    print("\n--- Starting data merge ---")

    print("--- Loading safe emails (Ling + Enron Ham) ---")

    ling_df = pd.read_csv(DATA_DIR / "Ling.csv")
    ling_df = ling_df[ling_df['label'] == 0][['body']].rename(columns={'body': 'Text'})

    enron_df = pd.read_csv(DATA_DIR / "enron_spam_data.csv")
    enron_df['Text'] = enron_df['Subject'].fillna('') + " " + enron_df['Message'].fillna('')
    enron_df = enron_df[enron_df['Spam/Ham'] == 'ham'][['Text']]

    safe_df = pd.concat([ling_df, enron_df], ignore_index=True)
    safe_df['Target'] = 0

    print("--- Loading Phishing attacks ---")

    nazario  = pd.read_csv(DATA_DIR / "Nazario.csv")[['body']].rename(columns={'body': 'Text'})
    nigerian = pd.read_csv(DATA_DIR / "Nigerian_Fraud.csv")[['body']].rename(columns={'body': 'Text'})
    ceas     = pd.read_csv(DATA_DIR / "CEAS_08.csv")
    ceas     = ceas[ceas['label'] == 1][['body']].rename(columns={'body': 'Text'})

    phishing_df = pd.concat([nazario, nigerian, ceas], ignore_index=True)
    phishing_df['Target'] = 1

    print("--- Loading SQLi attacks ---")

    sqli_df = pd.read_csv(SQLI_DATA, on_bad_lines='skip')
    sqli_df['Label'] = pd.to_numeric(sqli_df['Label'], errors='coerce')
    sqli_df = sqli_df[sqli_df['Label'] == 1]
    sqli_df = sqli_df[['Sentence']].rename(columns={'Sentence': 'Text'})
    sqli_df['Target'] = 2

    df_raw = pd.concat([safe_df, phishing_df, sqli_df], ignore_index=True)
    df_raw = df_raw.dropna(subset=['Text'])
    df_raw = df_raw[df_raw['Text'].str.strip() != '']

    print("\n--- Raw class distribution (before balancing) ---")
    print(df_raw['Target'].value_counts())

    counts      = df_raw['Target'].value_counts()
    min_count   = counts.min()
    target_n    = min(min_count, 15_000)  

    balanced_parts = []
    for cls in df_raw['Target'].unique():
        cls_df = df_raw[df_raw['Target'] == cls]
        if len(cls_df) > target_n:
            cls_df = resample(
                cls_df,
                n_samples=target_n,
                replace=False,
                random_state=42
            )
        balanced_parts.append(cls_df)

    df_balanced = pd.concat(balanced_parts, ignore_index=True)

    print("\n--- Balanced class distribution ---")
    print(df_balanced['Target'].value_counts())

    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    df = load_and_merge_data()
    print(df['Target'].value_counts())