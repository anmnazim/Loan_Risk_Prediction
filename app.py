import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Page Configuration & Custom Styling
# ==========================================
st.set_page_config(
    page_title="Loan Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Asset Loading (Cached)
# ==========================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('loan_rf_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, model_columns
    except Exception as e:
        st.error(f"⚠️ Error loading model assets: {e}")
        return None, None

model, model_columns = load_assets()

# ==========================================
# 3. Sidebar - User Inputs
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
st.sidebar.title("Applicant Profile")
st.sidebar.info("Enter details to assess loan eligibility.")

def get_user_inputs():
    with st.sidebar:
        st.subheader("📊 Numeric Details")
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=15000, step=500)
        credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=720)
        experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5, step=1)

        st.divider()
        st.subheader("📂 Categorical Details")
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education Level", ["Bachelor", "High School", "Masters", "PhD"])
        city = st.selectbox("City", ["Chicago", "Houston", "New York", "San Francisco"])
        employment = st.selectbox("Employment Type", ["Full-time", "Self-Employed", "Unemployed"])

    data = {
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'YearsExperience': experience,
        'Gender': gender,
        'Education': education,
        'City': city,
        'EmploymentType': employment
    }
    return pd.DataFrame([data])

# ==========================================
# 4. Main Application Interface
# ==========================================
st.title("🏦 Loan Risk Prediction Analysis")
st.write("Leveraging Machine Learning to evaluate loan applications instantly.")

if model is None:
    st.warning("Model not found. Please ensure `loan_rf_model.pkl` and `model_columns.pkl` are in the project folder.")
else:
    input_df = get_user_inputs()
    
    st.subheader("Input Summary")
    st.dataframe(input_df, hide_index=True)

    if st.button("Analyze Application"):
        with st.spinner("Processing data..."):
            # A. Preprocessing (Handling Dummies & Order)
            # Create dummies for the input
            df_encoded = pd.get_dummies(input_df)
            
            # Reconstruct the feature set to match model's expected columns
            # This handles drop_first=True byproduct and ensures order
            final_features = pd.DataFrame(columns=model_columns)
            
            # Fill common columns from input_df (numeric)
            for col in ['Age', 'Income', 'LoanAmount', 'CreditScore', 'YearsExperience']:
                final_features.loc[0, col] = input_df.loc[0, col]
            
            # Fill dummy columns (Gender_Male, City_Houston, etc.)
            for col in model_columns:
                if col in df_encoded.columns:
                    final_features.loc[0, col] = df_encoded.loc[0, col]
                elif '_' in col: # It's a dummy column not present in this specific input
                    final_features.loc[0, col] = 0
            
            # Ensure everything is numeric (float/int)
            final_features = final_features.fillna(0).astype(float)

            # B. Prediction
            prediction = model.predict(final_features)[0]
            probability = model.predict_proba(final_features)[0][1]

            # C. Display Results
            st.divider()
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown("### 🎯 Decision")
                if prediction == 1:
                    st.success("## APPROVED")
                    st.write("Congratulations! The applicant is likely to repay the loan.")
                else:
                    st.error("## REJECTED")
                    st.write("High risk detected. The loan application is rejected based on profile analysis.")
                
                st.metric("Approval Probability", f"{probability:.2%}")

            with res_col2:
                st.markdown("### 📈 Risk Score")
                st.progress(1 - probability)
                if probability > 0.8:
                    st.write("Risk Status: **Very Low**")
                elif probability > 0.6:
                    st.write("Risk Status: **Low**")
                elif probability > 0.4:
                    st.write("Risk Status: **Medium**")
                else:
                    st.write("Risk Status: **High**")

            # D. Feature Importance Visualization
            st.divider()
            st.subheader("🔍 Model Insight: Feature Importance")
            try:
                importances = model.feature_importances_
                feat_importances = pd.Series(importances, index=model_columns).sort_values(ascending=True).tail(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=feat_importances.values, y=feat_importances.index, ax=ax, palette="coolwarm")
                ax.set_title("Top 10 Factors Influencing the Decision")
                ax.set_xlabel("Importance Score")
                st.pyplot(fig)
            except:
                st.info("Feature importance chart not available for this model type.")

# Footer
st.markdown("---")
st.caption("© 2024 Loan Risk Analytics | Built with Python, Scikit-Learn, and Streamlit")
