import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt

# --- 1. Load Assets ---
@st.cache_resource
def load_model():
    return joblib.load('xgb_churn_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('sample_customers.csv')

model = load_model()
data = load_data()

# --- 2. UI Setup ---
st.title("📊 Transparent Churn Predictor")
st.markdown("Predict customer churn, understand *why*, and get actionable retention strategies.")

# Select a customer
customer_index = st.selectbox("Select a Customer Profile (Index):", data.index)
customer_data = data.iloc[[customer_index]]

if st.button("Analyze Customer Risk"):
    # --- 3. Prediction ---
    prob = model.predict_proba(customer_data)[0][1]
    risk_score = round(prob * 100, 2)
    
    st.subheader("Risk Assessment")
    if risk_score > 50:
        st.error(f"High Flight Risk: {risk_score}% probability of churning.")
    else:
        st.success(f"Low Flight Risk: {risk_score}% probability of churning.")

    # --- 4. SHAP Explainability ---
    st.subheader("Why is this happening?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(customer_data)
    
    # Plot Waterfall
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # --- 5. The "Hook": Recommendation Engine ---
    st.subheader("Actionable Recommendations")
    
    feature_names = customer_data.columns
    shap_vals_for_cust = shap_values.values[0]
    
    impacts = sorted(zip(feature_names, shap_vals_for_cust), key=lambda x: x[1], reverse=True)
    top_driver = impacts[0]
    
    if top_driver[1] > 0: 
        if top_driver[0] == 'MonthlyCharges':
            st.info("💡 **Action:** Cost is the primary driver. Offer a 3-month, 15% discount or bundle a free streaming service to increase perceived value.")
        elif top_driver[0] == 'tenure':
            st.info("💡 **Action:** Low tenure risk. Enroll them in the 'New Customer Success' onboarding email sequence.")
        elif top_driver[0] == 'InternetService':
            st.info("💡 **Action:** Competitors might be offering better fiber speeds. Offer a free speed upgrade trial.")
        else:
            st.info(f"💡 **Action:** The main issue is {top_driver[0]}. Have an account manager reach out specifically to address this pain point.")
    else:
        st.success("Customer is stable. No immediate retention action required. Consider upsell opportunities.")
