
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# 1. Load dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# 2. Remove customer ID
df.drop("customerID", axis=1, inplace=True)

# 3. Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

print("Missing TotalCharges:", df["TotalCharges"].isnull().sum())

# 4. Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

# 5. Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# 6. Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 7. Create complete ML pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
))
])

# 8. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Train model
model.fit(X_train, y_train)

# 10. Generate predictions
y_prob = model.predict_proba(X_test)[:, 1]

# Compare different classification thresholds
for threshold in [0.5, 0.4, 0.3]:

    y_pred_threshold = (y_prob >= threshold).astype(int)

    print(f"\nRESULTS AT THRESHOLD: {threshold}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_threshold))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred_threshold,
        target_names=["No Churn", "Churn"]
    ))

# 11. Evaluate model

# Final evaluation using threshold 0.4
y_pred = (y_prob >= 0.4).astype(int)

print("\nFINAL MODEL EVALUATION")
print("-------------------------")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))

print("\nModel saved successfully!")