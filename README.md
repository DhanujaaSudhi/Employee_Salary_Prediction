# 💼 Employee_Salary_Prediction  
https://employeesalaryprediction-7nlyrpvslaa94appcb9gauh.streamlit.app/

This project is a machine learning web app that predicts whether an employee’s annual income is <=50K or >50K based on demographic and work-related features.
It is built using Python, Streamlit, Scikit-learn, and XGBoost.

🚀 Features
1.Upload a CSV dataset (like the Adult Census Income dataset).
2.Data cleaning & preprocessing:
Handle missing values and unknown categories.
Drop irrelevant/duplicate features.
Encode categorical variables with OneHotEncoder.
Scale numerical variables with StandardScaler.
3.Train an XGBoost Classifier pipeline.
4.Display model accuracy score on test data.
5.Visualize:
Salary distribution by Age Group.
Salary distribution by Gender.
6.Upload new employee data for batch predictions and download results as CSV.

🧠 Machine Learning Workflow
1.Data Cleaning
Replace missing values, remove “Never-worked”/“Without-pay”, and drop duplicate columns.
2.Feature Engineering
Numerical features → StandardScaler
Categorical features → OneHotEncoder
3.Model Training
Train/Test Split (80/20).
Classifier: XGBoost (best performing).
4.Evaluation
Accuracy Score
Classification Report 
Confusion Matrix.

