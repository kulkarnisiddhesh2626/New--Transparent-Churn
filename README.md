# 📊 Enterprise Customer Churn Predictor & Retention Simulator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-orange)

**Live Demo:** [Click Here to View the Streamlit Dashboard](YOUR_STREAMLIT_APP_LINK_HERE)

## 🎯 Problem Statement
Customer retention is significantly cheaper than customer acquisition. However, while traditional Machine Learning models can predict *if* a customer will churn, they fail to explain *why*. This acts as a "black box" that business stakeholders cannot use to create targeted retention strategies.

## 💡 The Solution
This project bridges the gap between Data Science and Business Strategy. It deploys an **XGBoost** machine learning model wrapped in a **SHAP (SHapley Additive exPlanations)** explainability layer, hosted on a highly interactive **Streamlit** web application. 

Instead of just outputting a churn probability, the application provides a transparent, mathematical breakdown of the exact factors driving that risk, allowing account managers to deploy targeted, cost-effective interventions.

## 🚀 Key Features
* **🔍 Individual Risk Profiler:** Select a customer and instantly view their churn probability alongside a localized SHAP waterfall plot explaining the exact features driving their risk.
* **🎛️ What-If Retention Simulator:** Interactive sliders allow stakeholders to adjust a customer's monthly charges or tenure in real-time, simulating how specific discounts or contract extensions would impact their flight risk.
* **🌍 Global Insights:** Aggregated SHAP summary plots reveal the overarching drivers of churn across the entire business.
* **💡 Automated Recommendations:** A rule-based engine that reads the SHAP values and suggests concrete business actions (e.g., "Offer a 15% discount," "Enroll in onboarding").

## 🛠️ Architecture & Tech Stack
* **Data Processing & Engineering:** Pandas, NumPy, Scikit-Learn
* **Predictive Modeling:** XGBoost (Extreme Gradient Boosting Classifier)
* **Model Explainability:** SHAP (Game Theory)
* **Frontend Web Application:** Streamlit
* **Deployment:** Streamlit Community Cloud

## 🧠 Feature Engineering Highlight
To capture the psychological concept of "value for money," a custom feature `Tenure_to_Monthly_Ratio` was engineered. This successfully helped the model identify long-term customers who were quietly paying high premiums and were thus high-flight-risks for cheaper competitors.

## 👨‍💻 Run It Locally
Want to spin this up on your own machine?

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/transparent-churn.git](https://github.com/YOUR_USERNAME/transparent-churn.git)

# Navigate into the directory
cd transparent-churn

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
