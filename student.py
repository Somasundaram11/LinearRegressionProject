import pandas as pd
import numpy as np
import pickle
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "student_exam_score_dataset.csv")

df = pd.read_csv(csv_path)

print(df.columns)  # run once, then you can remove

# Features & target (safe way)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler, imputer), f)

print("Model trained and saved as model.pkl")
