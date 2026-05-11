import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

print("[INFO] Loading models...")

model = joblib.load("models/ckd_model.joblib")
scaler = joblib.load("models/scaler.joblib")
imputer = joblib.load("models/imputer.joblib")

encoded_columns = joblib.load("models/encoded_columns.joblib")
selected_features = joblib.load("models/selected_features.joblib")

# ================= DATA =================
df = pd.read_csv("dataset/kidney_disease_dataset.csv")

if "Target" in df.columns:
    df = df.drop("Target", axis=1)

# ================= FEATURE ENGINEERING =================
def safe_div(a, b):
    return a / (b + 1e-5)

if {'Serum creatinine (mg/dl)', 'Hemoglobin level (gms)'}.issubset(df.columns):
    df["Creatinine_Hemoglobin"] = df['Serum creatinine (mg/dl)'] * df['Hemoglobin level (gms)']

if {'Blood urea (mg/dl)', 'Serum creatinine (mg/dl)'}.issubset(df.columns):
    df["BUN_Creatinine_Ratio"] = safe_div(df['Blood urea (mg/dl)'], df['Serum creatinine (mg/dl)'])

# ================= ENCODE =================
df = pd.get_dummies(df)

# ALIGN
df = df.reindex(columns=encoded_columns, fill_value=0)

# IMPUTE
df = pd.DataFrame(imputer.transform(df), columns=encoded_columns)

# SCALE
df = pd.DataFrame(scaler.transform(df), columns=encoded_columns)

# FEATURE SELECT
df = df[selected_features]

print("[INFO] Final shape:", df.shape)

# SAVE
os.makedirs("shap_plots", exist_ok=True)
df.to_csv("shap_plots/X_sample.csv", index=False)

# RUN SHAP
print("\n[INFO] Running SHAP...")

for name in model.named_estimators_.keys():
    print(f"[INFO] Explaining {name}")

    result = os.system(f"python explain_single.py {name}")

print("\n✅ SHAP DONE")