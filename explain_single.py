import sys
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

from catboost import Pool

warnings.filterwarnings("ignore")

MODEL_PATH = "models/ckd_model.joblib"
model_name = sys.argv[1]
print(f"[INFO] Explaining model: {model_name}")

try:
    model = joblib.load(MODEL_PATH)
    base_model = model.named_estimators_[model_name]


    X_sample = pd.read_csv("shap_plots/X_sample.csv")
    print("[INFO] Data shape:", X_sample.shape)

    # ================= SHAP LOGIC =================
    if model_name == "cat":
        print("[INFO] Using CatBoost native SHAP (FINAL FIXED)...")

        pool = Pool(X_sample)

        shap_values = base_model.get_feature_importance(
            data=pool,
            type="ShapValues"
        )

        # 🔥 IMPORTANT: Handle multiclass FIRST
        if len(shap_values.shape) == 3:
            print("[INFO] Multiclass detected → selecting class 0")
            shap_values = shap_values[:, 0, :]   # (samples, features+1)

        # 🔥 THEN remove base value
        shap_values = shap_values[:, :-1]        # (samples, features)

        print("[INFO] Final SHAP shape:", shap_values.shape)

        # safety check
        if shap_values.shape[1] != X_sample.shape[1]:
            raise ValueError(
                f"Mismatch after fix → SHAP={shap_values.shape}, DATA={X_sample.shape}"
            )

    elif model_name in ["lgbm", "xgb"]:
        print(f"[INFO] Using TreeExplainer for {model_name.upper()}...")

        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_sample)

        # multiclass handling
        if isinstance(shap_values, list):
            print("[INFO] Multiclass detected → selecting class 0")
            shap_values = shap_values[0]

    else:
        raise ValueError("Unsupported model type")

    # ================= PLOT =================
    os.makedirs("shap_plots", exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)

    plt.title(f"SHAP Summary - {model_name.upper()}")

    output_path = f"shap_plots/shap_summary_{model_name}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[OK] Saved: {output_path}")

except Exception as e:
    print(f"[ERROR] SHAP failed for model '{model_name}': {e}")