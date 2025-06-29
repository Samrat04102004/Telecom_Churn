import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')


@st.cache_data
def load_data():
    return pd.read_csv('telco_processed_for_streamlit.csv')
df = load_data()

st.title('Customer Churn Prediction & Lifetime Value Dashboard')

# --- User Input Section ---
st.header('Input Customer Features')

# Gender
gender = st.selectbox('Gender', ['Male', 'Female'])

# Senior Citizen as Yes/No
senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])

# Partner
partner = st.selectbox('Partner', ['No', 'Yes'])

# Dependents
dependents = st.selectbox('Dependents', ['No', 'Yes'])

# Tenure
tenure = st.slider('Tenure (months)', 0, 72, 12)

# Phone Service
phone_service = st.selectbox('Phone Service', ['No', 'Yes'])

# Multiple Lines
multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])

# Internet Service
internet_service = st.selectbox('Internet Service', ['No', 'DSL', 'Fiber optic'])

# Online Security
online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])

# Online Backup
online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])

# Device Protection
device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])

# Tech Support
tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])

# Streaming TV
streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])

# Streaming Movies
streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])

# Contract
contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])

# Paperless Billing
paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])

# Payment Method
payment_method = st.selectbox('Payment Method', [
    'Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'
])

# Monthly Charges
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=70.0)

# Total Charges
total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=monthly_charges * tenure)

# --- Build input dictionary with all required features ---
input_dict = {}

# Binary features
input_dict['gender'] = 1 if gender == 'Female' else 0
input_dict['SeniorCitizen'] = 1 if senior_citizen == 'Yes' else 0
input_dict['Partner'] = 1 if partner == 'Yes' else 0
input_dict['Dependents'] = 1 if dependents == 'Yes' else 0
input_dict['PhoneService'] = 1 if phone_service == 'Yes' else 0
input_dict['PaperlessBilling'] = 1 if paperless_billing == 'Yes' else 0

# Numeric features
input_dict['tenure'] = tenure
input_dict['MonthlyCharges'] = monthly_charges
input_dict['TotalCharges'] = total_charges

# One-hot encoded features
# MultipleLines
input_dict['MultipleLines_No'] = 1 if multiple_lines == 'No' else 0
input_dict['MultipleLines_Yes'] = 1 if multiple_lines == 'Yes' else 0
input_dict['MultipleLines_No phone service'] = 1 if multiple_lines == 'No phone service' else 0

# InternetService
input_dict['InternetService_No'] = 1 if internet_service == 'No' else 0
input_dict['InternetService_DSL'] = 1 if internet_service == 'DSL' else 0
input_dict['InternetService_Fiber optic'] = 1 if internet_service == 'Fiber optic' else 0

# OnlineSecurity
input_dict['OnlineSecurity_No'] = 1 if online_security == 'No' else 0
input_dict['OnlineSecurity_Yes'] = 1 if online_security == 'Yes' else 0

# OnlineBackup
input_dict['OnlineBackup_No'] = 1 if online_backup == 'No' else 0
input_dict['OnlineBackup_Yes'] = 1 if online_backup == 'Yes' else 0

# DeviceProtection
input_dict['DeviceProtection_No'] = 1 if device_protection == 'No' else 0
input_dict['DeviceProtection_Yes'] = 1 if device_protection == 'Yes' else 0

# TechSupport
input_dict['TechSupport_No'] = 1 if tech_support == 'No' else 0
input_dict['TechSupport_Yes'] = 1 if tech_support == 'Yes' else 0

# StreamingTV
input_dict['StreamingTV_No'] = 1 if streaming_tv == 'No' else 0
input_dict['StreamingTV_Yes'] = 1 if streaming_tv == 'Yes' else 0

# StreamingMovies
input_dict['StreamingMovies_No'] = 1 if streaming_movies == 'No' else 0
input_dict['StreamingMovies_Yes'] = 1 if streaming_movies == 'Yes' else 0

# Contract
input_dict['Contract_Month-to-month'] = 1 if contract == 'Month-to-month' else 0
input_dict['Contract_One year'] = 1 if contract == 'One year' else 0
input_dict['Contract_Two year'] = 1 if contract == 'Two year' else 0

# PaymentMethod
input_dict['PaymentMethod_Bank transfer (automatic)'] = 1 if payment_method == 'Bank transfer (automatic)' else 0
input_dict['PaymentMethod_Credit card (automatic)'] = 1 if payment_method == 'Credit card (automatic)' else 0
input_dict['PaymentMethod_Electronic check'] = 1 if payment_method == 'Electronic check' else 0
input_dict['PaymentMethod_Mailed check'] = 1 if payment_method == 'Mailed check' else 0

# Ensure all features are present
for feat in feature_columns:
    if feat not in input_dict:
        input_dict[feat] = 0

input_df = pd.DataFrame([input_dict])[feature_columns]

# --- Dynamically calculate ServiceCount ---
service_features = [
    'PhoneService', 'InternetService_DSL', 'InternetService_Fiber optic',
    'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
    'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes'
]
input_df['ActiveInternetService'] = input_df[['InternetService_DSL', 'InternetService_Fiber optic']].max(axis=1)
input_df['ServiceCount'] = input_df['PhoneService'] + input_df['ActiveInternetService'] + input_df[service_features[3:]].sum(axis=1)

# --- Predict button ---
if st.button('Predict'):
    input_scaled = scaler.transform(input_df)
    proba = model.predict_proba(input_scaled)[0]
    st.subheader('Predicted Churn Probability')
    st.write(f"Probability of Retention: **{proba[0]:.2f}**")
    st.write(f"Probability of Churn: **{proba[1]:.2f}**")



    # --- Cohort Analysis ---
    st.header('Cohort Analysis')
    cohort_counts = df.groupby('tenure_group').size()
    cohort_churn = df.groupby('tenure_group')['Churn'].mean()

    fig1, ax1 = plt.subplots()
    cohort_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Customer Counts by Tenure Cohort')
    ax1.set_ylabel('Number of Customers')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    cohort_churn.plot(kind='bar', ax=ax2)
    ax2.set_title('Churn Rate by Tenure Cohort')
    ax2.set_ylabel('Churn Rate')
    st.pyplot(fig2)

    # --- RFM Segmentation ---
    st.header('RFM Segmentation')
    rfm = df[['tenure', 'ServiceCount', 'TotalCharges']].copy()
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Segment'] = kmeans.fit_predict(rfm)

    segment_names = {
        0: 'Established Mid-Value',
        1: 'Long-term High-Value',
        2: 'New or At-Risk Low-Value',
        3: 'Loyal Multi-Service Power Users'
    }
    rfm['SegmentName'] = rfm['Segment'].map(segment_names)

    st.write(rfm.groupby('SegmentName')[['Recency', 'Frequency', 'Monetary']].mean().round(2))
    st.write(rfm['SegmentName'].value_counts())

    fig3, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.boxplot(x='SegmentName', y='Recency', data=rfm, ax=axes[0])
    axes[0].set_title('Recency (Tenure) by Segment')
    axes[0].tick_params(axis='x', rotation=15)
    sns.boxplot(x='SegmentName', y='Frequency', data=rfm, ax=axes[1])
    axes[1].set_title('Frequency (Service Count) by Segment')
    axes[1].tick_params(axis='x', rotation=15)
    sns.boxplot(x='SegmentName', y='Monetary', data=rfm, ax=axes[2])
    axes[2].set_title('Monetary (Total Charges) by Segment')
    axes[2].tick_params(axis='x', rotation=15)
    plt.tight_layout()
    st.pyplot(fig3)

# --- Download Data Section ---
st.header('Download Processed Data')
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(label='Download CSV', data=csv, file_name='telco_processed.csv', mime='text/csv')

# --- Footer ---
st.markdown("""
---
**Project by Samrat**  
This dashboard demonstrates customer churn prediction,cohort analysis and customer segmentation using real-world data and industry-standard analytics.
""")
