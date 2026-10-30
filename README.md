# Customer Churn Prediction & Analytics Dashboard

A machine learning application that predicts customer churn using Logistic Regression and provides an interactive Streamlit dashboard for customer analytics, risk assessment, model explainability, and batch predictions.

## Overview

Customer churn occurs when customers discontinue a company's products or services. Identifying customers who are likely to leave helps businesses take proactive retention measures.

This project uses historical telecommunications customer data to train a machine learning classification model that estimates the probability of customer churn.

The application provides individual and batch predictions through an interactive dashboard, along with visual analytics and model interpretation.

## Key Features

- **Customer Churn Prediction:** Predicts whether an individual customer is likely to leave.
- **Churn Probability:** Displays the estimated probability of customer churn.
- **Customer Risk Classification:** Categorizes customers into Low, Medium, and High risk levels.
- **Business Analytics:** Displays total customers, churned customers, retained customers, and overall churn rate.
- **Data Visualization:** Visualizes churn distribution and churn patterns across contract types.
- **Model Explainability:** Uses Logistic Regression coefficients to identify features associated with higher and lower churn.
- **Batch Prediction:** Accepts CSV files containing multiple customer records.
- **Prediction Report:** Allows users to download batch prediction results as a CSV file.
- **Retention Recommendations:** Provides rule-based suggestions based on customer characteristics.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Algorithm | Logistic Regression |
| Data Visualization | Matplotlib, Streamlit |
| Model Persistence | Joblib |
| Development Environment | VS Code, Python Virtual Environment |
| Version Control | Git, GitHub |

## Dataset

The project uses the IBM Telco Customer Churn dataset, which contains customer information and service details.

- Total records: 7,043
- Total original columns: 21
- Target variable: `Churn`
- Churned customers: 1,869
- Customers who did not churn: 5,174

The dataset includes customer demographics, service subscriptions, contract details, payment methods, tenure, and billing information.

The `customerID` column is excluded from model training because it is an identifier rather than a predictive customer feature.

**Dataset source:** [IBM Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Machine Learning Workflow

1. **Data Exploration:** Examined dataset structure, feature types, distributions, and target variable.
2. **Data Preprocessing:** Converted `TotalCharges` to numeric format, handled missing values, and separated numerical and categorical features.
3. **Feature Engineering and Transformation:** Applied median imputation and standard scaling to numerical features, and most-frequent imputation and one-hot encoding to categorical features.
4. **Model Training:** Trained a Logistic Regression classifier using an 80:20 train-test split.
5. **Class Imbalance Handling:** Used `class_weight="balanced"` to account for the difference between churned and non-churned customers.
6. **Model Evaluation:** Evaluated performance using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
7. **Threshold Tuning:** Compared classification thresholds of 0.5, 0.4, and 0.3 and selected 0.4 as the initial operating threshold.

The preprocessing steps and classifier are combined into a Scikit-learn Pipeline and saved using Joblib.

## Model Performance

The following results were obtained using the balanced Logistic Regression model.

**Performance at the selected classification threshold of 0.4:**

| Metric | Result |
|---|---:|
| Accuracy | 70.05% |
| Churn Precision | 47% |
| Churn Recall | 87% |
| Churn F1-score | 61% |
| ROC-AUC | 0.8413 |

**Confusion Matrix:**

| | Predicted: No Churn | Predicted: Churn |
|---|---:|---:|
| Actual: No Churn | 664 | 371 |
| Actual: Churn | 50 | 324 |

The 0.4 threshold was selected as an initial compromise to improve churn detection.

It identifies more actual churn customers than the default 0.5 threshold, although this comes with an increase in false-positive predictions.

## Project Structure

```text
CustomerChurnPrediction/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── sample_customers.csv
│
├── model/
│   └── churn_model.pkl
│
├── app.py
├── explore_data.py
├── visualize_data.py
├── train_model.py
├── explain_model.py
├── test_batch.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** Dataset CSV files are excluded from Git tracking. Download the dataset
