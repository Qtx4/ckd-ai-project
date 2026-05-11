import joblib

print("[INFO] Loading saved features...")

# Direct load (BEST method)
selected_features = joblib.load("models/selected_features.joblib")

print(f"✅ Total features used in model: {len(selected_features)}")
print("\nSample features:", selected_features[:10])

# Optional: save copy (if needed)
joblib.dump(selected_features, "models/final_features.joblib")

print("\n🔥 Feature extraction DONE (SAFE + CORRECT)")