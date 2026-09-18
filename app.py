import streamlit as st
import pandas as pd
import numpy as np
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
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@st.cache_data
def get_demo_data():
    """A small set of realistic-looking sample transactions shown by default,
    so the app has something to demo immediately without requiring an upload."""
    rng = np.random.default_rng(42)
    n_rows = 10
    data = {"Time": rng.integers(0, 172792, n_rows)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1.5, n_rows)
    data["Amount"] = np.round(rng.exponential(50, n_rows), 2)
    return pd.DataFrame(data)


def predict(data: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict(data[FEATURE_COLUMNS])
    probs = model.predict_proba(data[FEATURE_COLUMNS])[:, 1]
    result = data.copy()
    result["Fraud Prediction"] = preds
    result["Fraud Probability"] = probs.round(4)
    return result


tab1, tab2, tab3 = st.tabs(["📊 Demo Data", "📁 Upload CSV", "✍️ Manual Entry"])

# ---------------- Tab 1: Default demo data (shown immediately) ----------------
with tab1:
    st.write("A sample of transactions, scored automatically — no upload needed.")
    demo_df = get_demo_data()
    demo_result = predict(demo_df)
    st.write(f"🚨 Predicted fraud cases: {int(demo_result['Fraud Prediction'].sum())} / {len(demo_result)}")
    st.dataframe(
        demo_result.style.applymap(
            lambda v: "background-color: #ffcccc" if v == 1 else "", subset=["Fraud Prediction"]
        )
    )

# ---------------- Tab 2: CSV upload ----------------
with tab2:
    st.write("Upload a CSV of transactions (same format as the training data: Time, V1-V28, Amount).")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if "Class" in data.columns:
            data = data.drop("Class", axis=1)

        result = predict(data)
        st.write(f"Processed {len(result)} transactions.")
        st.write(f"🚨 Predicted fraud cases: {int(result['Fraud Prediction'].sum())}")
        st.dataframe(
            result[["Fraud Prediction", "Fraud Probability"]].style.applymap(
                lambda v: "background-color: #ffcccc" if v == 1 else "", subset=["Fraud Prediction"]
            )
        )
    else:
        st.info("👆 Upload a CSV file to see predictions here.")

# ---------------- Tab 3: Manual input ----------------
with tab3:
    st.write("Enter a transaction's details manually to get an instant prediction.")
    st.caption(
        "V1-V28 are anonymized (PCA-transformed) features from the original dataset. "
        "For a quick test, leave them at 0 and just adjust Time/Amount, or use "
        "'Randomize V1-V28' to simulate a realistic transaction."
    )

    col1, col2 = st.columns(2)
    with col1:
        manual_time = st.number_input("Time (seconds since first transaction)", min_value=0, value=50000)
    with col2:
        manual_amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=1.0)

    randomize = st.checkbox("Randomize V1-V28 (simulate a realistic transaction)", value=True)

    if randomize:
        rng = np.random.default_rng()
        v_values = rng.normal(0, 1.5, 28)
    else:
        v_values = np.zeros(28)

    if st.button("Predict this transaction"):
        row = {"Time": manual_time}
        for i, v in enumerate(v_values, start=1):
            row[f"V{i}"] = v
        row["Amount"] = manual_amount

        manual_df = pd.DataFrame([row])[FEATURE_COLUMNS]
        result = predict(manual_df)

        pred = int(result["Fraud Prediction"].iloc[0])
        prob = float(result["Fraud Probability"].iloc[0])

        if pred == 1:
            st.error(f"🚨 Predicted: FRAUD (probability: {prob:.2%})")
        else:
            st.success(f"✅ Predicted: Legitimate (fraud probability: {prob:.2%})")

st.divider()
st.caption(
    "Model: Random Forest (100 trees) trained on SMOTE-balanced data | "
    "ROC-AUC: 0.96 | Built by Tripti Sharma"
)
