
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_model.joblib"
DATA_PATH = BASE_DIR / "data" / "phishing_dataset.csv"

st.set_page_config(
    page_title="Phishing URL Risk Analyzer",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Phishing URL Risk Analyzer")
st.write(
    "Machine-learning-based phishing website classification "
    "using the UCI Phishing Websites dataset."
)

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Required files are missing. Run python train_model.py first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_names = bundle["feature_names"]

df = pd.read_csv(DATA_PATH)

st.success(f"Loaded model: {bundle['model_name']}")

col1, col2, col3 = st.columns(3)
col1.metric("Dataset records", f"{len(df):,}")
col2.metric("Input features", len(feature_names))
col3.metric("Reported test accuracy", "97.42%")

st.divider()
st.subheader("Dataset-based prediction")

st.info(
    "This demonstration predicts from an existing dataset record. "
    "It does not yet analyze a newly entered raw URL."
)

record_number = st.number_input(
    "Select a dataset record number",
    min_value=1,
    max_value=len(df),
    value=1,
    step=1,
)

record = df.iloc[int(record_number) - 1]
actual_label = int(record["target"])
features = record[feature_names].to_frame().T

if st.button("Analyze selected record", type="primary"):
    prediction = int(model.predict(features)[0])

    if prediction == -1:
        st.error("Model prediction: PHISHING WEBSITE")
    elif prediction == 1:
        st.success("Model prediction: LEGITIMATE WEBSITE")
    else:
        st.warning(f"Model returned an unexpected label: {prediction}")

    actual_text = (
        "Phishing website" if actual_label == -1
        else "Legitimate website"
    )
    predicted_text = (
        "Phishing website" if prediction == -1
        else "Legitimate website"
    )

    st.write(f"**Dataset's actual label:** {actual_text}")
    st.write(f"**Model's prediction:** {predicted_text}")

    if prediction == actual_label:
        st.success("Prediction matches the dataset label.")
    else:
        st.warning("Prediction differs from the dataset label.")

    st.caption(
        "This selected record may belong to the model's training data, "
        "so this result is not an independent accuracy measurement."
    )

with st.expander("View the selected record's 30 features"):
    st.dataframe(features.T.rename(columns={features.index[0]: "Value"}))

st.divider()
st.subheader("Model comparison")

comparison_path = DATA_PATH.parent / "model_comparison.csv"

if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)
    st.bar_chart(
        comparison.set_index("Model")["Test Accuracy (%)"]
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)

st.caption(
    "Educational project only. Predictions can be wrong; "
    "do not rely on this prototype alone for security decisions."
)
