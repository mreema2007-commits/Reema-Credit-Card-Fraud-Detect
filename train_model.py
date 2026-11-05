# ==========================================
# CREDIT CARD FRAUD DETECTION
# STEP 1: LOAD AND TRAIN MODELS
# ==========================================

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

train_path = "data/fraudTrain.csv"
test_path = "data/fraudTest.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print("Training dataset shape:", train_data.shape)
print("Testing dataset shape:", test_data.shape)


# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "amt",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long"
]

target = "is_fraud"

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

print("\nFeatures selected:")
print(features)


# ==========================================
# 3. SCALE DATA
# ==========================================

print("\nScaling data...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data scaling completed!")


# ==========================================
# 4. LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

logistic_model.fit(X_train_scaled, y_train)

lr_pred = logistic_model.predict(X_test_scaled)
lr_probability = logistic_model.predict_proba(X_test_scaled)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred, zero_division=0)
lr_recall = recall_score(y_test, lr_pred, zero_division=0)
lr_f1 = f1_score(y_test, lr_pred, zero_division=0)
lr_roc_auc = roc_auc_score(y_test, lr_probability)


# ==========================================
# 5. DECISION TREE
# ==========================================

print("Training Decision Tree...")

decision_tree = DecisionTreeClassifier(
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

decision_tree.fit(X_train_scaled, y_train)

dt_pred = decision_tree.predict(X_test_scaled)
dt_probability = decision_tree.predict_proba(X_test_scaled)[:, 1]

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred, zero_division=0)
dt_recall = recall_score(y_test, dt_pred, zero_division=0)
dt_f1 = f1_score(y_test, dt_pred, zero_division=0)
dt_roc_auc = roc_auc_score(y_test, dt_probability)


# ==========================================
# 6. RANDOM FOREST
# ==========================================

print("Training Random Forest...")

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train_scaled, y_train)

rf_pred = random_forest.predict(X_test_scaled)
rf_probability = random_forest.predict_proba(X_test_scaled)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, zero_division=0)
rf_recall = recall_score(y_test, rf_pred, zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
rf_roc_auc = roc_auc_score(y_test, rf_probability)


# ==========================================
# 7. MODEL COMPARISON
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        lr_accuracy,
        dt_accuracy,
        rf_accuracy
    ],
    "Precision": [
        lr_precision,
        dt_precision,
        rf_precision
    ],
    "Recall": [
        lr_recall,
        dt_recall,
        rf_recall
    ],
    "F1 Score": [
        lr_f1,
        dt_f1,
        rf_f1
    ],
    "ROC-AUC": [
        lr_roc_auc,
        dt_roc_auc,
        rf_roc_auc
    ]
})

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(results.to_string(index=False))


# ==========================================
# 8. SELECT RANDOM FOREST
# ==========================================

print("\n==========================================")
print("FINAL MODEL")
print("==========================================")

print("Selected Model: Random Forest")


# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

print("\nRandom Forest Confusion Matrix:")

print(confusion_matrix(y_test, rf_pred))


# ==========================================
# 10. CREATE MODELS FOLDER
# ==========================================

os.makedirs("models", exist_ok=True)


# ==========================================
# 11. SAVE RANDOM FOREST MODEL
# ==========================================

joblib.dump(random_forest, "models/fraud_model.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

print("\n==========================================")
print("MODEL SAVED SUCCESSFULLY")
print("==========================================")

print("Saved:")
print("models/fraud_model.pkl")
print("models/scaler.pkl")

print("\nProject training completed successfully!")