# 📊 Telecom Customer Churn Prediction & Analytics Project

## 🚀 Project Overview
This repository showcases an <strong>end-to-end machine learning pipeline</strong> for:

- Predicting <strong>customer churn</strong> in the telecom sector 
- Segmenting customers using <strong>cohort analysis</strong> and <strong>RFM segmentation</strong>

It includes advanced feature engineering, robust model evaluation, and a <strong>Streamlit dashboard</strong> for interactive insights.

---

## 🔍 Key Features

- **Customer Churn Prediction**  
  Built and evaluated multiple ML models using real-world telecom data.

- **Feature Engineering**  
  Created features like `ServiceCount`, `TenureGroup`, and `ContractType`.

- **Model Evaluation**  
  Metrics include ROC-AUC, F1-score, precision, and recall.

- **Cohort Analysis**  
  Evaluated churn trends across customer tenure groups.

- **RFM Segmentation**  
  Applied Recency-Frequency-Monetary segmentation using KMeans clustering.

- **Interactive Streamlit App**  
  Accepts user input and displays churn risk, CLV, and customer segment.

---

## 🛠️ Technologies & Libraries

- **Language**: Python 3.13  
- **Libraries**:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`
  - `lifelines`
  - `imblearn` (SMOTE)
  - `optuna`
  - `Streamlit`

---

## 📁 Project Structure

Telecom_Churn/

├── feature_columns.pkl # Feature column order for prediction

├── logistic_regression_model.pkl # Trained ML model

├── main.py # Streamlit dashboard script

├── requirements.txt # Dependency list

├── scaler.pkl # Trained scaler for input preprocessing

├── telco_processed_for_streamlit.csv # Cleaned and encoded dataset

└── main.ipynb # Jupyter notebook with full analysis

---

## 📊 Model Development

- **Preprocessing**: Cleaned and one-hot encoded the data (`drop_first=False`)
- **Feature Engineering**: Added `ActiveInternetService`, `ServiceCount`
- **Class Imbalance**: Applied SMOTE
- **Models Trained**: Logistic Regression, Random Forest, SVM, Gradient Boosting
- **Evaluation**: Used ROC-AUC, classification reports, confusion matrices
- **Hyperparameter Tuning**: Done using Optuna
- **Threshold Optimization**: Chosen to maximize F1-score

---

## 📈 Business Insights

- Customers with **long tenure** and **multiple services** are less likely to churn.
- **Month-to-month contracts** correlate strongly with higher churn.
- **Active Internet Service** leads to significantly lower churn.
- **RFM segmentation** helps tailor customer engagement strategies.

---

## 🖥️ Streamlit App Features

- 🎛️ Easy-to-use dropdowns and sliders to input customer attributes  
- 📊 Live prediction of churn probabilities  
- 💰 Estimated Customer Lifetime Value (CLV)  
- 🧠 Segment insights via Cohort & RFM analysis  
- 📥 Option to download the processed dataset  

---

## ⭐ Acknowledgments

- IBM Telco Customer Churn Dataset
- Open-source Python libraries including:


> Thank you for exploring this project!

---

