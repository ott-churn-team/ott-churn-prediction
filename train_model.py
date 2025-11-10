import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import time

print("\n🚀 Starting Advanced Customer Churn Model Training...\n")
start_time = time.time()

# ------------------------------
# 1️⃣ LOAD DATASET
# ------------------------------
data_path = os.path.join(os.path.dirname(__file__), "customer_data.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError("❌ Dataset not found! Please keep 'customer_data.csv' in this folder.")

data = pd.read_csv(data_path)
print(f"✅ Dataset loaded successfully! ({data.shape[0]} rows)")
print("Columns:", list(data.columns), "\n")

# ------------------------------
# 2️⃣ CLEAN & PREPROCESS
# ------------------------------
if "Churn" in data.columns:
    data.rename(columns={"Churn": "churn"}, inplace=True)

# Fix churn labels
data["churn"] = data["churn"].astype(str).str.strip().str.lower().map({
    "yes": 1, "no": 0, "y": 1, "n": 0, "1": 1, "0": 0
}).fillna(0).astype(int)

# Drop irrelevant columns
drop_cols = ["customer_id", "phone_no", "year"]
data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore", inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = le.fit_transform(data[col].astype(str))

# Fill missing numeric values
data = data.fillna(data.median(numeric_only=True))

# ------------------------------
# 3️⃣ BALANCE CLASSES (force even samples)
# ------------------------------
churned = data[data["churn"] == 1]
not_churned = data[data["churn"] == 0]

if len(churned) == 0:
    # If no churned customers exist, make some pseudo-churn examples
    churned = not_churned.sample(min(10, len(not_churned)//5)).copy()
    churned["churn"] = 1
    print("⚠️ Added synthetic churn samples for training...")

# Make both classes equal
min_len = min(len(churned), len(not_churned))
data_balanced = pd.concat([
    churned.sample(min_len, replace=True, random_state=42),
    not_churned.sample(min_len, replace=True, random_state=42)
]).sample(frac=1, random_state=42)

X = data_balanced.drop(columns=["churn"])
y = data_balanced["churn"]

print(f"✅ Balanced dataset: {len(X)} records ({y.sum()} churn / {len(y)-y.sum()} non-churn)\n")

# ------------------------------
# 4️⃣ TRAIN/TEST SPLIT & SCALE
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 5️⃣ TRAIN MODEL
# ------------------------------
model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced",
    max_depth=None,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
print("✅ Model trained successfully!\n")

# ------------------------------
# 6️⃣ EVALUATE
# ------------------------------
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"📊 Accuracy: {acc*100:.2f}%")
print("🧾 Classification Report:\n", classification_report(y_test, y_pred))
print("🔢 Predictions:", np.bincount(y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Customer Churn")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ------------------------------
# 7️⃣ FEATURE IMPORTANCE
# ------------------------------
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=importances.head(10), y=importances.head(10).index, palette="viridis")
plt.title("Top 10 Important Features Influencing Churn")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# ------------------------------
# 8️⃣ SAVE MODEL & SCALER
# ------------------------------
os.makedirs("models", exist_ok=True)
with open("models/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ------------------------------
# ✅ SUMMARY
# ------------------------------
end_time = time.time()
print(f"\n🎯 Model & Scaler saved in /models/")
print(f"Training Time: {end_time - start_time:.2f}s")
print("✨ Training Complete! Ready for Streamlit deployment.")
