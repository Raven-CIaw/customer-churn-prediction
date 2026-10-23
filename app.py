
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "model.pkl is not a trained prediction model."
        )

    return model, scaler


st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to estimate their likelihood of churning."
)

try:
    model, scaler = load_artifacts()

    with st.form("customer_form"):
        age = st.slider("Age", 18, 65, 35)
        gender = st.selectbox("Gender", ["Female", "Male"])
        tenure = st.slider("Tenure (months)", 1, 60, 24)
        usage = st.slider("Usage Frequency", 1, 30, 15)
        support_calls = st.slider("Support Calls", 0, 10, 2)
        payment_delay = st.slider("Payment Delay (days)", 0, 30, 5)

        subscription = st.selectbox(
            "Subscription Type",
            ["Basic", "Standard", "Premium"]
        )
        contract = st.selectbox(
            "Contract Length",
            ["Quarterly", "Monthly", "Annual"]
        )

        total_spend = st.slider(
            "Total Spend", 100, 1000, 600
        )
        last_interaction = st.slider(
            "Days Since Last Interaction", 1, 30, 10
        )

        submitted = st.form_submit_button(
            "Predict Churn",
            type="primary"
        )

    if submitted:
        subscription_map = {
            "Basic": 0,
            "Standard": 1,
            "Premium": 2
        }
        contract_map = {
            "Quarterly": 0,
            "Monthly": 1,
            "Annual": 2
        }

        features = pd.DataFrame([{
            "Age": age,
            "Tenure": tenure,
            "Usage Frequency": usage,
            "Support Calls": support_calls,
            "Payment Delay": payment_delay,
            "Subscription Type": subscription_map[subscription],
            "Contract Length": contract_map[contract],
            "Total Spend": total_spend,
            "Last Interaction": last_interaction,
            "Gender_Female": int(gender == "Female"),
            "Gender_Male": int(gender == "Male")
        }])

        # Match the exact feature order used in training.
        features = features[scaler.feature_names_in_]
        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        churn_index = list(model.classes_).index(1)
        churn_probability = probabilities[churn_index]

        st.subheader("Prediction Result")
        st.metric(
            "Estimated churn probability",
            f"{churn_probability:.1%}"
        )

        if prediction == 1:
            st.warning("The model predicts that this customer may churn.")
        else:
            st.success("The model predicts that this customer may stay.")

        st.caption(
            "This is a model estimate, not a guarantee of customer behavior."
        )

except Exception as e:
    st.error(f"Could not load or run the model: {e}")