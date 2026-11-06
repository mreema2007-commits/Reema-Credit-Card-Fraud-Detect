import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Reema Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ============================================
# LOAD MODEL
# ============================================

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# ============================================
# HEADER
# ============================================

st.title("💳 Credit Card Fraud Detection System")

st.write(
    "A Machine Learning based application for detecting "
    "potentially fraudulent credit card transactions."
)

st.divider()

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("📌 Project Menu")

page = st.sidebar.radio(
    "Select Page",
    [
        "🔍 Fraud Prediction",
        "📊 Model Analysis",
        "ℹ️ About Project"
    ]
)

# ============================================
# PAGE 1 - FRAUD PREDICTION
# ============================================

if page == "🔍 Fraud Prediction":

    st.header("🔍 Transaction Fraud Prediction")

    st.write(
        "Enter the transaction details below."
    )

    col1, col2 = st.columns(2)

    with col1:

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0,
            step=10.0
        )

        latitude = st.number_input(
            "Customer Latitude",
            value=17.70
        )

        longitude = st.number_input(
            "Customer Longitude",
            value=83.30
        )

    with col2:

        city_population = st.number_input(
            "City Population",
            min_value=0,
            value=100000
        )

        merchant_latitude = st.number_input(
            "Merchant Latitude",
            value=17.75
        )

        merchant_longitude = st.number_input(
            "Merchant Longitude",
            value=83.35
        )

    st.divider()

    if st.button(
        "🔎 Analyze Transaction",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "amt": [amount],
            "lat": [latitude],
            "long": [longitude],
            "city_pop": [city_population],
            "merch_lat": [merchant_latitude],
            "merch_long": [merchant_longitude]
        })

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)[0]

        # Probability
        probability = model.predict_proba(
            input_scaled
        )[0][1]

        probability_percent = probability * 100

        st.divider()

        st.header("📊 Detection Result")

        # ------------------------------------
        # FRAUD RESULT
        # ------------------------------------

        if prediction == 1:

            st.error(
                "🚨 POTENTIAL FRAUD DETECTED"
            )

            risk = "HIGH RISK"

        else:

            st.success(
                "✅ NORMAL TRANSACTION"
            )

            if probability < 0.20:
                risk = "LOW RISK"

            elif probability < 0.50:
                risk = "MEDIUM RISK"

            else:
                risk = "HIGH RISK"

        # ------------------------------------
        # PROBABILITY
        # ------------------------------------

        st.metric(
            "Fraud Probability",
            f"{probability_percent:.2f}%"
        )

        st.progress(
            min(float(probability), 1.0)
        )

        st.info(
            f"Risk Level: **{risk}**"
        )

        # ------------------------------------
        # TRANSACTION SUMMARY
        # ------------------------------------

        st.subheader("📋 Transaction Summary")

        summary = pd.DataFrame({
            "Parameter": [
                "Transaction Amount",
                "Customer Latitude",
                "Customer Longitude",
                "City Population",
                "Merchant Latitude",
                "Merchant Longitude",
                "Fraud Probability",
                "Risk Level"
            ],

            "Value": [
                f"₹{amount:.2f}",
                latitude,
                longitude,
                city_population,
                merchant_latitude,
                merchant_longitude,
                f"{probability_percent:.2f}%",
                risk
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


# ============================================
# PAGE 2 - MODEL ANALYSIS
# ============================================

elif page == "📊 Model Analysis":

    st.header("📊 Random Forest Model Analysis")

    st.write(
        "Performance analysis of the trained Random Forest "
        "fraud detection model."
    )

    # -----------------------------------------
    # MODEL METRICS
    # -----------------------------------------

    st.subheader("🏆 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            "94.07%"
        )

    with col2:
        st.metric(
            "Precision",
            "4.86%"
        )

    with col3:
        st.metric(
            "Recall",
            "77.20%"
        )

    with col4:
        st.metric(
            "ROC-AUC",
            "94.61%"
        )

    st.divider()

    # -----------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------

    st.subheader("🔲 Confusion Matrix")

    confusion_matrix_path = Path(
        "report/confusion_matrix.png"
    )

    if confusion_matrix_path.exists():

        st.image(
            str(confusion_matrix_path),
            caption="Random Forest Confusion Matrix",
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion matrix image not found."
        )

    st.divider()

    # -----------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------

    st.subheader("📈 Feature Importance")

    feature_path = Path(
        "report/feature_importance.png"
    )

    if feature_path.exists():

        st.image(
            str(feature_path),
            caption="Random Forest Feature Importance",
            use_container_width=True
        )

    else:

        st.warning(
            "Feature importance image not found."
        )

    st.divider()

    # -----------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------

    st.subheader("🤖 Machine Learning Algorithm")

    st.write(
        "**Random Forest Classifier**"
    )

    st.write(
        "Random Forest combines multiple decision trees "
        "to improve prediction performance and reduce "
        "the risk of overfitting."
    )

    st.success(
        "Random Forest was selected as the final model "
        "because it achieved the highest ROC-AUC score "
        "among the three tested models."
    )
    # ============================================
# PAGE 3 - ABOUT PROJECT
# ============================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About Credit Card Fraud Detection")

    st.write(
        "This project uses Machine Learning to identify "
        "potentially fraudulent credit card transactions."
    )

    st.divider()

    # -----------------------------------------
    # OBJECTIVE
    # -----------------------------------------

    st.subheader("🎯 Project Objective")

    st.write(
        "The main objective of this project is to develop "
        "a machine learning system that can classify "
        "credit card transactions as normal or potentially fraudulent."
    )

    # -----------------------------------------
    # TECHNOLOGIES
    # -----------------------------------------

    st.subheader("🛠️ Technologies Used")

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Matplotlib",
            "Streamlit",
            "Joblib"
        ],

        "Purpose": [
            "Programming language",
            "Data processing",
            "Numerical operations",
            "Machine Learning",
            "Data visualization",
            "Web application",
            "Model saving and loading"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------
    # MACHINE LEARNING MODELS
    # -----------------------------------------

    st.subheader("🤖 Machine Learning Models")

    models = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],

        "Purpose": [
            "Baseline classification model",
            "Tree-based classification",
            "Ensemble classification model"
        ]
    })

    st.dataframe(
        models,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------
    # FINAL MODEL
    # -----------------------------------------

    st.subheader("🏆 Final Model")

    st.success(
        "Random Forest was selected as the final model "
        "because it achieved the highest ROC-AUC score "
        "among the tested models."
    )

    # -----------------------------------------
    # FEATURES
    # -----------------------------------------

    st.subheader("📋 Input Features")

    features = [
        "Transaction Amount",
        "Customer Latitude",
        "Customer Longitude",
        "City Population",
        "Merchant Latitude",
        "Merchant Longitude"
    ]

    for feature in features:
        st.write("•", feature)

    # -----------------------------------------
    # WORKFLOW
    # -----------------------------------------

    st.subheader("🔄 Project Workflow")

    st.write(
        "Dataset → Data Preprocessing → Feature Selection → "
        "Feature Scaling → Model Training → Model Evaluation → "
        "Random Forest Selection → Fraud Prediction"
    )

    # -----------------------------------------
    # STUDENT PROJECT
    # -----------------------------------------

    st.divider()

    st.info(
        "This application was developed as an academic "
        "Machine Learning project for credit card fraud detection."
    )
