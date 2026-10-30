
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("model/churn_model.pkl")

# Get preprocessing and classifier steps
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Get Logistic Regression coefficients
coefficients = classifier.coef_[0]

# Create a DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Sort by coefficient
importance_df = importance_df.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nTOP 10 FEATURES ASSOCIATED WITH HIGHER CHURN")
print(importance_df.head(10).to_string(index=False))

print("\nTOP 10 FEATURES ASSOCIATED WITH LOWER CHURN")
print(
    importance_df.tail(10)
    .sort_values(by="Coefficient")
    .to_string(index=False)
)
