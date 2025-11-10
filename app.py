import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="OTT Platform Management", page_icon="🎬", layout="wide")

# ------------------------------
# SESSION STATE
# ------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ------------------------------
# NAVIGATION HELPERS
# ------------------------------
def go_to(page):
    st.session_state.page = page
    st.rerun()


def check_login(username, password):
    if username == "admin" and password == "1234":
        st.session_state.logged_in = True
        st.success("✅ Login successful! Redirecting to dashboard...")
        st.balloons()
        go_to("dashboard")
    else:
        st.error("❌ Invalid username or password.")


# ------------------------------
# LOAD MODEL & SCALER
# ------------------------------
def load_model_and_scaler():
    model_path = os.path.join("models", "churn_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("⚠️ Model or Scaler not found! Please train the model first.")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


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


# ------------------------------
# CHURN DASHBOARD FUNCTION
# ------------------------------
def churn_dashboard():
    st.title("📊 Customer Churn Prediction Dashboard")

    model, scaler = load_model_and_scaler()
    st.success("✅ Model and Scaler loaded successfully!")

    training_features = [
        "gender", "age", "no_of_days_subscribed", "multi_screen", "mail_subscribed",
        "weekly_mins_watched", "minimum_daily_mins", "maximum_daily_mins",
        "weekly_max_night_mins", "videos_watched", "maximum_days_inactive",
        "customer_support_calls"
    ]

    st.sidebar.header("Choose Prediction Type")
    option = st.sidebar.radio("Select Option:", ["Upload Dataset", "Single Customer Prediction"])
    threshold = st.sidebar.slider("Churn Probability Threshold (%)", 0, 100, 50, 5) / 100

    # ------------------------------
    # UPLOAD DATASET OPTION
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

            # Pie Chart
            churn_labels = ["Likely to Churn", "Not Likely to Churn"]
            churn_values = [churn_count, total_customers - churn_count]
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            ax_pie.pie(churn_values, labels=churn_labels, autopct="%1.1f%%", startangle=90)
            ax_pie.axis("equal")
            st.pyplot(fig_pie)

            # Confusion Matrix
            y_test = np.array(data.get("Churn", np.zeros(len(data))))
            y_pred = predictions
            st.markdown("## 🟦 Model Performance & Insights")

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Churn", "Churn"])
            fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
            disp.plot(ax=ax_cm, cmap='Blues', colorbar=False)
            plt.title("Confusion Matrix - Customer Churn")
            st.pyplot(fig_cm)

            # Feature Importance
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                sorted_indices = np.argsort(importances)[::-1]
                fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
                ax_fi.bar(range(len(training_features)), importances[sorted_indices])
                ax_fi.set_xticks(range(len(training_features)))
                ax_fi.set_xticklabels(np.array(training_features)[sorted_indices], rotation=45, ha="right")
                ax_fi.set_title("Top Features Impacting Churn")
                st.pyplot(fig_fi)

            csv = original_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predicted Results",
                data=csv,
                file_name="predicted_churn_results.csv",
                mime="text/csv",
            )

    # ------------------------------
    # SINGLE CUSTOMER OPTION
    # ------------------------------
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

            # Confusion Matrix + Feature Importance
            y_test = np.array([prediction])
            y_pred = np.array([prediction])

            st.markdown("## 🟦 Model Performance & Insights")

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Churn", "Churn"])
            fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
            disp.plot(ax=ax_cm, cmap='Blues', colorbar=False)
            plt.title("Confusion Matrix - Customer Churn")
            st.pyplot(fig_cm)

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                sorted_indices = np.argsort(importances)[::-1]
                fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
                ax_fi.bar(range(len(training_features)), importances[sorted_indices])
                ax_fi.set_xticks(range(len(training_features)))
                ax_fi.set_xticklabels(np.array(training_features)[sorted_indices], rotation=45, ha="right")
                ax_fi.set_title("Top Features Impacting Churn")
                st.pyplot(fig_fi)


# ------------------------------
# PAGE LOGIC
# ------------------------------
if st.session_state.page == "home":
    st.title("🎬 OTT Platform Management System")
    st.subheader("Customer Churn Prediction Dashboard")
    st.write("""
        Welcome to the **OTT Management System**.  
        This platform helps analyze user data and predict churn likelihood.
    """)

    if not st.session_state.logged_in:
        st.info("🔒 Please log in to access the dashboard.")
        if st.button("Go to Login Page"):
            go_to("login")
    else:
        st.success("✅ Logged in as Admin.")
        if st.button("Go to Churn Prediction Dashboard"):
            go_to("dashboard")
        if st.button("Logout"):
            st.session_state.logged_in = False
            go_to("home")

elif st.session_state.page == "login":
    st.title("🔐 Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        check_login(username, password)
    if st.button("⬅️ Back to Home"):
        go_to("home")

elif st.session_state.page == "dashboard":
    churn_dashboard()












