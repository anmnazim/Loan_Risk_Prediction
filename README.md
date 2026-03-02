# Loan Risk Prediction App

A professional web application built with **Streamlit** that uses a **Random Forest Classifier** to predict loan approval risk. It evaluates applicants based on financial and personal data, providing real-time decisions and risk probabilities.

## 🚀 Features
- **Real-time Prediction**: Instantly check "Approved" or "Rejected" status.
- **Probability Scoring**: View the likelihood of approval.
- **Feature Importance**: Understand which factors shaped the model's decision.
- **Interactive UI**: User-friendly sidebar for data entry.
- **Clean Architecture**: Automatic handling of categorical encoding and feature ordering.

## 📂 Project Structure
```
Loan-Risk-Prediction/
├── app.py              # Main Streamlit application
├── loan_rf_model.pkl   # Trained Random Forest model
├── model_columns.pkl   # Expected feature columns for the model
├── requirements.txt    # Dependency list
└── README.md           # Documentation
```

## 🛠️ Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/loan-risk-prediction.git
   cd loan-risk-prediction
   ```

2. **Create a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## ☁️ Deployment

### Streamlit Cloud (Easiest)
1. Push your code to a Public GitHub repository.
2. Sign in to [Streamlit Cloud](https://share.streamlit.io/).
3. Click "New App" and select your repository.
4. Set the Main file path to `app.py`.
5. Click **Deploy**.

### Render
1. Connect your GitHub repository to [Render](https://render.com/).
2. Select **Web Service**.
3. Set Environment to `Python`.
4. Build Command: `pip install -r requirements.txt`.
5. Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## 📊 Model Details
- **Algorithm**: Random Forest Classifier
- **Metric**: ~0.96 Accuracy | ~0.94 ROC-AUC
- **Target**: Balanced class weights for improved minority class detection.

---
Developed as part of the Loan Risk Prediction Project.
