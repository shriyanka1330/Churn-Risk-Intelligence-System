# Predictive Modeling and Risk Scoring for Bank Customer Churn

## Overview

This project presents an explainable machine learning framework for predicting customer churn in retail banking. The system analyzes customer demographic, financial, and behavioral data to identify customers who are at risk of leaving the bank. It also generates churn risk scores and provides explainable predictions using SHAP (SHapley Additive Explanations).

The project aims to help banks shift from reactive customer retention to proactive, data-driven decision-making.

---

## Features

* Customer churn prediction using Machine Learning
* Churn risk score generation
* Feature engineering for improved prediction
* Explainable AI using SHAP
* Interactive Streamlit web application
* Performance comparison of multiple machine learning models
* Data visualization and exploratory data analysis

---

## Machine Learning Models

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

Among these models, **Gradient Boosting** achieved the best overall performance with:

* Accuracy: **86.2%**
* ROC-AUC: **0.8652**

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib
* Seaborn
* Streamlit

---

## Dataset

* Bank Customer Churn Dataset
* 10,000 customer records
* Customer demographic, financial, and behavioral attributes

Target Variable:

* **Exited = 1** → Customer Churned
* **Exited = 0** → Customer Retained

---

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. SHAP Explainability
8. Customer Risk Prediction
9. Streamlit Dashboard

---

## Project Structure

```text
├── data/
├── notebooks/
├── models/
├── app.py
├── requirements.txt
├── README.md
└── assets/
```

---

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Results

The proposed system accurately predicts customer churn using ensemble machine learning models. SHAP analysis provides transparent explanations by identifying the most influential features affecting customer churn, including customer age, number of products, active membership status, geographical location, and balance-to-salary ratio.

---

## Future Work

* Real-time customer churn prediction
* Deep learning-based models
* Cloud deployment
* Integration with live banking systems
* Automated retention recommendation engine

---

## Author

**Shriyanka Bhardwaj**
