import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load Data
train_df = pd.read_csv("heart_disease_preprocessing/train.csv")
test_df  = pd.read_csv("heart_disease_preprocessing/test.csv")

TARGET = "target"
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# MLflow Setup
mlflow.set_experiment("heart-disease-baseline")

# Training Autolog
with mlflow.start_run(run_name="baseline-random-forest"):
    mlflow.sklearn.autolog() 

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print("Run selesai. Cek MLflow UI di http://127.0.0.1:5000")