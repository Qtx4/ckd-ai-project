import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, StackingClassifier

from imblearn.combine import SMOTETomek
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna

# ================= LOAD =================
df = pd.read_csv("dataset/kidney_disease_dataset.csv")
df.replace(['?', ' ', 'NA', 'nan'], np.nan, inplace=True)

# ================= TARGET =================
le = LabelEncoder()
df["Target"] = le.fit_transform(df["Target"].astype(str))

# ================= FEATURE ENGINEERING =================
def safe_div(a, b):
    return a / (b + 1e-5)

if {'Serum creatinine (mg/dl)', 'Hemoglobin level (gms)'}.issubset(df.columns):
    df["Creatinine_Hemoglobin"] = df['Serum creatinine (mg/dl)'] * df['Hemoglobin level (gms)']

if {'Blood urea (mg/dl)', 'Serum creatinine (mg/dl)'}.issubset(df.columns):
    df["BUN_Creatinine_Ratio"] = safe_div(df['Blood urea (mg/dl)'], df['Serum creatinine (mg/dl)'])

# ================= SPLIT =================
X = df.drop("Target", axis=1)
y = df["Target"]

# ================= ENCODING =================
cat_cols = X.select_dtypes(include=["object", "string"]).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

encoded_columns = X.columns.tolist()

# ================= IMPUTER =================
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=encoded_columns)

# ================= SCALER =================
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=encoded_columns)

# ================= OUTLIER REMOVAL =================
iso = IsolationForest(contamination=0.02, random_state=42)
mask = iso.fit_predict(X) != -1
X, y = X[mask], y[mask]

# ================= FEATURE SELECTION =================
selector = SelectKBest(mutual_info_classif, k=min(25, X.shape[1]))
X_sel = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
X = pd.DataFrame(X_sel, columns=selected_features)

# ================= SMOTE =================
smote = SMOTETomek(random_state=42)
X, y = smote.fit_resample(X, y)

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= OPTUNA =================
def objective(trial):
    params = {
        "depth": trial.suggest_int("depth", 6, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
        "iterations": trial.suggest_int("iterations", 200, 350),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 5),
        "verbose": 0
    }

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    scores = []

    for tr, val in skf.split(X_train, y_train):
        model = CatBoostClassifier(**params)
        model.fit(X_train.iloc[tr], y_train.iloc[tr])
        preds = model.predict(X_train.iloc[val])
        scores.append(accuracy_score(y_train.iloc[val], preds))

    return np.mean(scores)

print("🧠 Optuna tuning...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

best_params = study.best_params
best_params["verbose"] = 0

# ================= MODELS =================
cat = CatBoostClassifier(**best_params)
cat.fit(X_train, y_train)

lgbm = LGBMClassifier(n_estimators=150, random_state=42, n_jobs=-1)
lgbm.fit(X_train, y_train)

xgb = XGBClassifier(
    n_estimators=150,
    tree_method="hist",
    verbosity=0,
    random_state=42
)
xgb.fit(X_train, y_train)

# ================= STACK =================
stack = StackingClassifier(
    estimators=[("cat", cat), ("lgbm", lgbm), ("xgb", xgb)],
    final_estimator=LGBMClassifier(n_estimators=100),
    n_jobs=-1
)

stack.fit(X_train, y_train)

# ================= EVAL =================
preds = stack.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, preds))

# ================= SAVE (CLEAN) =================
os.makedirs("models", exist_ok=True)

joblib.dump(stack, "models/ckd_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(imputer, "models/imputer.joblib")
joblib.dump(le, "models/encoder.joblib")

joblib.dump(encoded_columns, "models/encoded_columns.joblib")
joblib.dump(selected_features.tolist(), "models/selected_features.joblib")
joblib.dump(list(X.columns), "models/final_features.joblib")

print("\n🔥 TRAINING COMPLETE (PRO LEVEL)")