import pandas as pd

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display first five rows
print("FIRST FIVE ROWS:")
print(df.head())

# Dataset dimensions
print("\nDATASET SHAPE:")
print(df.shape)

# Column information
print("\nDATASET INFORMATION:")
print(df.info())

# Missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Churn distribution
print("\nCHURN DISTRIBUTION:")
print(df["Churn"].value_counts())