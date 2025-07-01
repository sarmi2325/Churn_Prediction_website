import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import shap
import json
import matplotlib.pyplot as plt

# Load saved model and features
with open('model2.pkl', 'rb') as f:
    model = cloudpickle.load(f)
with open('selected_features2.json') as f:
    selected_features = json.load(f)

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📊 Telecom Customer Churn Prediction")

# Sidebar input method
input_mode = st.sidebar.radio("Choose Input Method", ("Manual Input", "Upload CSV"))
st.sidebar.code("\n".join(selected_features), language="python")

# 🔧 Add churn threshold slider
threshold = st.sidebar.slider("Set Churn Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Define feature input form
def get_user_input():
    st.header("Customer Info")
    input_data = {
        'MonthlyCharges': st.number_input('Monthly Charges', 0, 200, 70),
        'TotalCharges': st.number_input('Total Charges', 0, 10000, 2000),
        'tenure': st.number_input('Tenure (months)', 0, 72, 12),
        'is_new_customer': st.selectbox('Is New Customer?', [0, 1]),
        'InternetService_Fiber optic': st.selectbox('Fiber Optic?', [0, 1]),
        'Contract_Two year': st.selectbox('Two Year Contract?', [0, 1]),
        'Contract_One year': st.selectbox('One Year Contract?', [0, 1]),
        'StreamingMovies_Yes': st.selectbox('Streaming Movies?', [0, 1]),
        'StreamingTV_Yes': st.selectbox('Streaming TV?', [0, 1]),
        'PaymentMethod_Electronic check': st.selectbox('Electronic Check?', [0, 1]),
    }
    return pd.DataFrame([input_data])

# Prediction logic
def predict(data):
    preds = model.predict(data)
    proba = model.predict_proba(data)[:, 1]
    return preds, proba

# SHAP explanation logic
def explain_model(data):
    explainer = shap.Explainer(model.named_estimators_['lgbm'])
    shap_values = explainer(data)

    st.subheader("🔍 SHAP Summary Plot")
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0], ax=ax, show=False)
    st.pyplot(fig)

# Load input data
if input_mode == "Manual Input":
    user_df = get_user_input()
    process_batch = False
else:
    file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    if file is not None:
        user_df = pd.read_csv(file)
        process_batch = True
    else:
        user_df = None
        process_batch = False

# Align input features
if user_df is not None:
    try:
        X_input = user_df[selected_features]
        predictions, probabilities = predict(X_input)
        prediction_flags = (probabilities >= threshold).astype(int)

        if process_batch:
            user_df['Churn Probability'] = probabilities
            user_df['Churn Prediction'] = prediction_flags

            st.subheader("📥 Predictions Added to CSV")
            st.dataframe(user_df.head())

            # Download link
            csv_download = user_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📩 Download Predictions CSV",
                data=csv_download,
                file_name="churn_predictions.csv",
                mime='text/csv'
            )
        else:
            st.subheader("📈 Prediction Result")
            st.write(f"**Churn Threshold:** {threshold:.2f}")
            st.write(f"**Churn Probability:** {probabilities[0]:.2f}")
            st.write(f"**Prediction:** {'Churn' if prediction_flags[0] == 1 else 'No Churn'}")

            if probabilities[0] >= threshold + 0.15:
                st.error("🚨 High Risk: Take Immediate Action")
            elif probabilities[0] >= threshold:
                st.warning("⚠️ Medium Risk")
            else:
                st.success("✅ Low Risk")

            explain_model(X_input)

    except Exception as e:
        st.error(f"⚠️ Error in prediction: {e}")

