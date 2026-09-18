import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write(
    "This app uses a Random Forest model trained on 284,000+ real credit card "
    "transactions (highly imbalanced data, balanced using SMOTE) to predict whether "
    "a transaction is fraudulent."
)

st.divider()

model = joblib.load("fraud_model.pkl")

st.subheader("Try it yourself")
st.write("Upload a CSV of transactions (same format as the training data), or use the sample below.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if "Class" in data.columns:
        data = data.drop("Class", axis=1)

    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    data["Fraud Prediction"] = predictions
    data["Fraud Probability"] = probabilities.round(4)

    st.write(f"Processed {len(data)} transactions.")
    st.write(f"🚨 Predicted fraud cases: {int(predictions.sum())}")

    st.dataframe(
        data[["Fraud Prediction", "Fraud Probability"]].style.applymap(
            lambda v: "background-color: #ffcccc" if v == 1 else "", subset=["Fraud Prediction"]
        )
    )
else:
    st.info("👆 Upload a CSV file to see predictions here.")

st.divider()
st.caption(
    "Model: Random Forest (100 trees) trained on SMOTE-balanced data | "
    "ROC-AUC: 0.96 | Built by Tripti Sharma"
)
