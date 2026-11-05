import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# -----------------------------------------
# Load dataset
# -----------------------------------------

print("Loading test dataset...")

test_data = pd.read_csv("data/fraudTest.csv")

features = [
    "amt",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long"
]

X_test = test_data[features]
y_test = test_data["is_fraud"]

# -----------------------------------------
# Load trained model
# -----------------------------------------

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Scale test data
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# Prediction
# -----------------------------------------

y_pred = model.predict(X_test_scaled)

# -----------------------------------------
# Classification Report
# -----------------------------------------

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Fraud"],
        zero_division=0
    )
)

# -----------------------------------------
# Confusion Matrix
# -----------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)

# -----------------------------------------
# Display Confusion Matrix
# -----------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fraud"]
)

display.plot()

plt.title("Random Forest Confusion Matrix")

plt.savefig(
    "report/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved to:")
print("report/confusion_matrix.png")

# -----------------------------------------
# Feature Importance
# -----------------------------------------

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

print(feature_importance)

# -----------------------------------------
# Feature Importance Graph
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "report/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFeature importance graph saved to:")
print("report/feature_importance.png")

print("\nAnalysis completed successfully!")