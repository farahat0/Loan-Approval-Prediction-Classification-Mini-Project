# Loan Approval Prediction: Classification Mini-Project

## 📌 Overview
This project applies data mining classification techniques to predict whether a loan application will be approved based on applicant financial and demographic data. The objective is to clean, explore, and model the data using Decision Trees and Logistic Regression.

## 🛠️ Project Steps

* **1. Data Loading & Inspection**
  * Loaded `loan_data.csv` using Pandas.
  * Evaluated data types, unique values, and missing values.
  * Dropped the `BankruptcyHistory` column as it contained only one unique value across the entire dataset and added no predictive power.

* **2. Data Cleaning & Formatting**
  * Standardized the text capitalization in the target variable, `LoanApproved`.
  * Cleaned the `LoanDuration` column by removing the string ' months' and converting the values to pure integers.
  * Cleaned currency columns (`AnnualIncome`, `LoanAmount`, `MonthlyLoanPayment`, `MonthlyIncome`) by stripping '$' and ',' characters and converting them to numeric types.

* **3. Handling Missing Values**
  * Imputed missing `EmploymentStatus` categorical values using the most frequent value (mode).
  * Logically calculated missing `MonthlyIncome` values by dividing the applicant's `AnnualIncome` by 12.
  * Calculated missing `MonthlyLoanPayment` values by dividing the `LoanAmount` by the `LoanDuration`.

* **4. Machine Learning Modeling**
  * Trained a **Decision Tree Classifier** to establish rule-based predictions based on conditions like Income and Credit Score.
  * Built a **Logistic Regression** model (using the 'liblinear' solver) as a secondary classification approach.
  * Evaluated the models comprehensively using Accuracy, Precision, Recall, F1-Score, and Normalized Confusion Matrices.
  * Applied **Cross-Validation** to compare the performance of the default models against hyperparameter-tuned versions to see if further optimizations were effective.

## 🚀 Technologies Used
* **Data Processing:** Python, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Decision Trees, Logistic Regression)
