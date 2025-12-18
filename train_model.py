import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("student_exam_score_dataset.csv")

# Features & target
X = df[['Hours_Studied']]
y = df['Score']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model, scaler, imputer
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler, imputer), f)

print("Model saved as model.pkl")
