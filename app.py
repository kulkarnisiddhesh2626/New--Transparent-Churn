import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt

# --- 1. Page Config & Load Assets ---
st.set_page_config(page_title="Enterprise Churn Predictor", layout="wide", page_icon="📈")

@st.cache_resource
def load_model():
    return joblib.load('xgb_churn_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('sample_customers.csv')

model = load_model()
data = load_data()

# --- 2. Sidebar: Author & Tech Stack (Dropdowns) ---
with st.sidebar:
    st.title("Navigation & Info")
    
    with st.expander("👨‍💻 About the Author", expanded=False):
        st.markdown("**Created by:** Siddhesh Kulkarni")
        st.markdown("📧 [kulkarnisiddhesh2626@gmail.com](mailto:kulkarnisiddhesh2626@gmail.com)")
        st.markdown("🔗 [LinkedIn Profile](https://www.linkedin.com/in/siddhesh-kulkarni-b2a600207/)")
        st.markdown("💻 [GitHub](https://github.com/)") # Update with your GitHub profile link if desired
        
    with st.expander("🛠️ Tech Stack", expanded=False):
        st.markdown("- **Language:** Python")
        st.markdown("- **Data Manipulation:** Pandas, NumPy")
        st.markdown("- **Modeling:** XGBoost (Gradient Boosted Trees)")
        st.markdown("- **Explainability:** SHAP (Game Theory)")
        st.markdown("- **Deployment:** Streamlit Community Cloud")

# --- 3. Main Header ---
st.title("📈 Enterprise Churn Predictor & Retention Simulator")
st.markdown("Empowering business stakeholders with explainable AI to predict customer churn and simulate retention strategies in real-time.")
st.markdown("---")

# --- 4. Documentation Section (Detailed Expanders) ---
st.markdown("### 📖 Project Documentation")

with st.expander("🎯 Problem Statement & Business Value"):
    st.markdown("""
    **The Challenge:** Businesses often know *which* customers are likely to leave, but struggle to understand *why*. Traditional machine learning models act as "black boxes," providing predictions without actionable context. 
    
    **The Solution:** This application bridges the gap between data science and business strategy. It not only predicts the probability of a customer churning but also provides a transparent, mathematical breakdown of the exact factors driving that risk. 
    
    **Business Impact:** By understanding the *why*, account managers can deploy targeted retention strategies (e.g., specific discounts, targeted onboarding) rather than relying on generic, costly blanket promotions.
    """)

with st.expander("🏗️ Project Architecture & Machine Learning Pipeline"):
    st.markdown("""
    **1. Data Processing & Feature Engineering:**
    * Ingests raw subscription data and encodes categorical variables for machine readability.
    * Engineers advanced business metrics, such as the `Tenure_to_Monthly_Ratio`, to capture 'value for money' sentiment and identify high-flight-risk demographics.
    
    **2. Predictive Modeling (The Brain):**
    * Utilizes **XGBoost (eXtreme Gradient Boosting)**, a state-of-the-art decision tree ensemble model known for high accuracy on complex, tabular data.
    
    **3. Explainability Layer (The Translator):**
    * Integrates **SHAP (SHapley Additive exPlanations)**, a game-theoretic approach to explain the output of any machine learning model.
    * Converts complex model weights into localized, human-readable waterfall plots to explain individual predictions.
    """)

with st.expander("📊 About the Dataset & Features"):
    st.markdown("""
    **Source:** Telco Customer Churn Dataset.
    
    **Key Feature Categories Analyzed by the Model:**
    * **Demographics:** Gender, Age, Dependents, Partner status.
    * **Account Information:** Tenure (months), Contract type (Month-to-month vs. Annual), Payment method.
    * **Service Subscriptions:** Internet service type (Fiber optic vs. DSL), Tech support, Streaming services.
    * **Financials:** Monthly charges, Total charges.
    
    **Custom Engineered Features:**
    * `Tenure_to_Monthly_Ratio`: Helps identify long-term customers paying high premiums.
    """)

with st.expander("📥 Data Preview & Download"):
    st.markdown("Interact with the sample test data used for these predictions. You can sort, scroll, and download the dataset for local analysis.")
    st.dataframe(data.head(100), use_container_width=True) 
    
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Test Dataset (CSV)",
        data=csv,
        file_name='churn_sample_data.csv',
        mime='text/csv',
    )

st.markdown("---")

# --- 5. Main Dashboard Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Individual Analysis", "🎛️ What-If Simulator", "🌍 Global Insights"])

# --- Tab 1: Original Individual Analysis ---
with tab1:
    st.write("### Customer Risk Profiler")
    customer_index = st.selectbox("Select a Customer Profile (Index):", data.index, key="tab1_select")
    customer_data = data.iloc[[customer_index]]

    if st.button("Analyze Customer Risk"):
        prob = model.predict_proba(customer_data)[0][1]
        risk_score = prob * 100
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Risk Assessment")
            if risk_score > 50:
                st.error(f"High Flight Risk: **{risk_score:.2f}%**")
            else:
                st.success(f"Low Flight Risk: **{risk_score:.2f}%**")
                
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(customer_data)
            impacts = sorted(zip(customer_data.columns, shap_values.values[0]), key=lambda x: x[1], reverse=True)
            top_driver = impacts[0]
            
            st.subheader("Actionable Recommendations")
            if top_driver[1] > 0: 
                if top_driver[0] == 'MonthlyCharges':
                    st.info("💡 **Action:** Cost is the primary driver. Offer a discount or bundle.")
                elif top_driver[0] == 'tenure':
                    st.info("💡 **Action:** Low tenure risk. Enroll in onboarding sequence.")
                else:
                    st.info(f"💡 **Action:** Main issue is {top_driver[0]}. Have an account manager reach out.")
            else:
                st.success("Customer is stable. Consider upsell opportunities.")

        with col2:
            st.subheader("Why is this happening?")
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

# --- Tab 2: The What-If Simulator ---
with tab2:
    st.write("### Retention Simulator")
    st.markdown("Adjust the levers below to see how changes in price or tenure impact this customer's churn risk.")
    
    sim_index = st.selectbox("Select a Customer to Simulate:", data.index, key="tab2_select")
    sim_data = data.iloc[[sim_index]].copy()
    
    base_prob = model.predict_proba(sim_data)[0][1] * 100
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        new_monthly = st.slider("Adjust Monthly Charges ($)", 
                                min_value=15.0, max_value=120.0, 
                                value=float(sim_data['MonthlyCharges'].values[0]))
        
        new_tenure = st.slider("Simulate Increased Tenure (Months)", 
                               min_value=0, max_value=72, 
                               value=int(sim_data['tenure'].values[0]))
        
        sim_data['MonthlyCharges'] = new_monthly
        sim_data['tenure'] = new_tenure
        sim_data['Tenure_to_Monthly_Ratio'] = sim_data['tenure'] / (sim_data['MonthlyCharges'] + 1)
        
    with col_sim2:
        new_prob = model.predict_proba(sim_data)[0][1] * 100
        diff = new_prob - base_prob
        
        st.metric(label="Simulated Churn Probability", 
                  value=f"{new_prob:.2f}%", 
                  delta=f"{diff:.2f}% (Change from Baseline)",
                  delta_color="inverse")
        
        st.markdown("*Note: Negative change (Green) means the customer is less likely to churn after your intervention.*")

# --- Tab 3: Global Insights ---
with tab3:
    st.write("### Global Feature Importance")
    st.markdown("What drives churn across the **entire** customer base?")
    
    if st.button("Generate Global SHAP Plot"):
        with st.spinner("Calculating overall impacts..."):
            explainer = shap.TreeExplainer(model)
            shap_values_global = explainer.shap_values(data.head(100)) 
            
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_values_global, data.head(100), show=False)
            st.pyplot(fig2)
