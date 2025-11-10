import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def churn_dashboard():
    # ------------------------------
    # PAGE CONFIG
    # ------------------------------
    st.title("📊 Customer Churn Prediction Dashboard")

    # ------------------------------
    # LOAD MODEL & SCALER
    # ------------------------------
    model_path = os.path.join("models", "churn_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("⚠️ Model or Scaler not found! Please train the model first using `train_model.py`.")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    st.success("✅ Model and Scaler loaded successfully!")

    # ------------------------------
    # FEATURES USED DURING TRAINING
    # ------------------------------
    training_features = [
        "gender", "age", "no_of_days_subscribed", "multi_screen", "mail_subscribed",
        "weekly_mins_watched", "minimum_daily_mins", "maximum_daily_mins",
        "weekly_max_night_mins", "videos_watched", "maximum_days_inactive",
        "customer_support_calls"
    ]

    # ------------------------------
    # SIDEBAR OPTIONS
    # ------------------------------
    st.sidebar.header("Choose Prediction Type")
    option = st.sidebar.radio("Select Option:", ["Upload Dataset", "Single Customer Prediction"])

    # ------------------------------
    # RETENTION STRATEGY FUNCTION
    # ------------------------------
    def generate_retention_strategy(prob):
        if prob >= 0.8:
            return "⚠️ High risk — Offer big discounts or free premium month."
        elif prob >= 0.6:
            return "💰 Medium-high risk — Send personalized offers or call retention team."
        elif prob >= 0.4:
            return "📩 Moderate risk — Engage via email reminders or app notifications."
        else:
            return "✅ Low risk — Customer seems satisfied. Maintain engagement."

    threshold = st.sidebar.slider("Churn Probability Threshold (%)", 0, 100, 50, 5) / 100

    # ------------------------------
    # OPTION 1 – UPLOAD DATASET
    # ------------------------------
    if option == "Upload Dataset":
        uploaded_file = st.file_uploader("📁 Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

        if uploaded_file:
            if uploaded_file.name.endswith(".xlsx"):
                data = pd.read_excel(uploaded_file)
            else:
                data = pd.read_csv(uploaded_file)

            st.write("### 📂 Uploaded Data Preview:")
            st.dataframe(data.head())

            original_data = data.copy()
            data = data.drop(columns=["Churn", "churn"], errors="ignore")

            data_encoded = data.copy()
            for col in data_encoded.select_dtypes(include=["object"]).columns:
                data_encoded[col] = pd.factorize(data_encoded[col])[0]

            data_encoded = data_encoded.fillna(0)
            data_encoded = data_encoded[training_features]
            scaled_data = scaler.transform(data_encoded)

            churn_probs = model.predict_proba(scaled_data)[:, 1]
            predictions = (churn_probs > threshold).astype(int)

            original_data["Churn Probability (%)"] = np.round(churn_probs * 100, 2)
            original_data["Predicted_Churn"] = np.where(predictions == 1, "Likely to Churn", "Not Likely to Churn")
            original_data["Retention Strategy"] = [generate_retention_strategy(p) for p in churn_probs]

            total_customers = len(original_data)
            churn_count = np.sum(predictions)
            st.markdown(f"### 📊 Summary:")
            st.markdown(
                f"Out of **{total_customers} customers**, **{churn_count} are likely to churn** and **{total_customers - churn_count} are not likely to churn.**"
            )

            def color_churn(val):
                if val == "Likely to Churn":
                    return "color: red; font-weight: bold"
                elif val == "Not Likely to Churn":
                    return "color: green; font-weight: bold"
                return ""

            st.write("### 🧾 Prediction Results:")
            st.dataframe(original_data.style.applymap(color_churn, subset=["Predicted_Churn"]))

            churn_labels = ["Likely to Churn", "Not Likely to Churn"]
            churn_values = [churn_count, total_customers - churn_count]
            st.write("### 📈 Churn Distribution:")
            st.pyplot(pd.Series(churn_values, index=churn_labels).plot.pie(
                autopct="%1.1f%%", figsize=(4, 4), ylabel=""
            ).get_figure())

    else:
        st.subheader("🎯 Enter Customer Details for Prediction")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 10, 100, 30)
        no_of_days_subscribed = st.number_input("No. of Days Subscribed", 0, 1000, 120)
        multi_screen = st.selectbox("Uses Multiple Screens?", ["Yes", "No"])
        mail_subscribed = st.selectbox("Subscribed to Email?", ["Yes", "No"])
        weekly_mins_watched = st.number_input("Weekly Minutes Watched", 0, 10000, 1500)
        minimum_daily_mins = st.number_input("Minimum Daily Minutes", 0, 500, 20)
        maximum_daily_mins = st.number_input("Maximum Daily Minutes", 0, 1000, 200)
        weekly_max_night_mins = st.number_input("Weekly Max Night Minutes", 0, 10000, 600)
        videos_watched = st.number_input("Videos Watched", 0, 1000, 50)
        maximum_days_inactive = st.number_input("Maximum Days Inactive", 0, 100, 5)
        customer_support_calls = st.number_input("Customer Support Calls", 0, 50, 2)

        if st.button("🔮 Predict Churn"):
            input_df = pd.DataFrame({
                "gender": [1 if gender == "Male" else 0],
                "age": [age],
                "no_of_days_subscribed": [no_of_days_subscribed],
                "multi_screen": [1 if multi_screen == "Yes" else 0],
                "mail_subscribed": [1 if mail_subscribed == "Yes" else 0],
                "weekly_mins_watched": [weekly_mins_watched],
                "minimum_daily_mins": [minimum_daily_mins],
                "maximum_daily_mins": [maximum_daily_mins],
                "weekly_max_night_mins": [weekly_max_night_mins],
                "videos_watched": [videos_watched],
                "maximum_days_inactive": [maximum_days_inactive],
                "customer_support_calls": [customer_support_calls]
            })

            scaled_input = scaler.transform(input_df)
            churn_prob = model.predict_proba(scaled_input)[0][1]
            prediction = 1 if churn_prob > threshold else 0

            st.markdown(f"### **Prediction:** {'🚨 Likely to Churn' if prediction == 1 else '✅ Not Likely to Churn'}")
            st.progress(float(churn_prob))
            st.markdown(f"**Churn Probability:** {churn_prob * 100:.2f}%")

            strategy = generate_retention_strategy(churn_prob)
            st.info(f"💡 **Suggested Retention Strategy:** {strategy}")

























