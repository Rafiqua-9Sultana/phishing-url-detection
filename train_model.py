
from pathlib import Path

import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Create project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

print("Downloading the UCI Phishing Websites dataset...")

# UCI Machine Learning Repository dataset ID: 327
dataset = fetch_ucirepo(id=327)

X = dataset.data.features
y = dataset.data.targets

if X is None or y is None:
    raise ValueError("Dataset features or targets could not be loaded.")

# Convert the target to a one-dimensional series
if isinstance(y, pd.DataFrame):
    if y.shape[1] != 1:
        raise ValueError("Unexpected number of target columns.")
    y = y.iloc[:, 0]

# Save a local copy for future use
combined = X.copy()
combined["target"] = y
combined.to_csv(DATA_DIR / "phishing_dataset.csv", index=False)

print(f"Dataset downloaded: {X.shape[0]} rows, {X.shape[1]} features")
print("Preparing training and testing data...")

# Keep a consistent split for fair model comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    ),
    "Decision Tree": make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(random_state=42)
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    )
}

results = []
best_name = None
best_model = None
best_accuracy = -1

print("\nTraining models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Model": name,
        "Test Accuracy (%)": round(accuracy * 100, 2)
    })

    print(f"{name}: {accuracy * 100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_name = name
        best_model = model

# Save comparison results
results_df = pd.DataFrame(results).sort_values(
    by="Test Accuracy (%)",
    ascending=False
)
results_df.to_csv(DATA_DIR / "model_comparison.csv", index=False)

# Save the selected model and its feature names
model_bundle = {
    "model": best_model,
    "model_name": best_name,
    "feature_names": list(X.columns),
    "target_classes": list(pd.Series(y).unique())
}

joblib.dump(model_bundle, MODEL_DIR / "phishing_model.joblib")

print("\nModel comparison:")
print(results_df.to_string(index=False))

print(f"\nBest model: {best_name}")
print(f"Saved model to: {MODEL_DIR / 'phishing_model.joblib'}")
print(f"Saved dataset to: {DATA_DIR / 'phishing_dataset.csv'}")
print("\nTraining completed successfully!")
print("\nClassification report for the best model:")
print(classification_report(
    y_test,
    best_model.predict(X_test),
    zero_division=0
))

print("Phishing URL Detection Project")
print("Python file is working correctly!")
