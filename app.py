import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Set page config
st.set_page_config(page_title="Employee Salary Prediction", layout="wide")

st.title("💼 Employee Salary Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file (like adult.csv):", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded. Using example dataset.")
    df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv", header=None)
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                  'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                  'hours-per-week', 'native-country', 'income']

# Data cleaning
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)
df = df[df['workclass'] != 'Without-pay']
df = df[df['workclass'] != 'Never-worked']
if 'education' in df.columns:
    df.drop(columns=['education'], inplace=True)

# Convert income to 0 and 1
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# Features and target
X = df.drop(columns='income')
y = df['income']

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Model pipeline
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(n_estimators=100, eval_metric="logloss"))
])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.success(f"✅ Model trained! Accuracy: {acc:.2f}")

# Prediction on uploaded data
st.subheader("🔍 Salary Distribution by Age & Gender")

# Age bins
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 60, 100],
                         labels=['<25', '25-35', '35-45', '45-60', '60+'])

fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.countplot(data=df, x='age_group', hue='income', palette="pastel", ax=ax1)
ax1.set_title("Salary Distribution by Age Group")
ax1.set_xlabel("Age Group")
ax1.set_ylabel("Count")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x='sex', hue='income', palette="muted", ax=ax2)
ax2.set_title("Salary Distribution by Gender")
ax2.set_xlabel("Gender")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# Optional: upload new user data for prediction
st.subheader("📂 Upload New Data for Salary Prediction")
new_file = st.file_uploader("Upload new employee data CSV (same columns as training data):", key="predict")

if new_file:
    new_data = pd.read_csv(new_file)
    if 'income' in new_data.columns:
        new_data.drop(columns='income', inplace=True)
    predictions = pipe.predict(new_data)
    new_data['Predicted Income'] = np.where(predictions == 1, '>50K', '<=50K')
    st.write(new_data.head())
    csv = new_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")
