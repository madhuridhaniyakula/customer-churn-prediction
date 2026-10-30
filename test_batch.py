
import pandas as pd

df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Select 20 customer records
sample = df.drop(columns=["customerID", "Churn"]).head(20)

sample.to_csv(
    "data/sample_customers.csv",
    index=False
)

print("Sample CSV created successfully!")
