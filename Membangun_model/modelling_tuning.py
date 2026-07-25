import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

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
mlflow.set_experiment("heart-disease-tuning")

# Hyperparameter Grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

# Training + Manual Logging
with mlflow.start_run(run_name="rf-gridsearch-tuning"):

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Manual Log: Params
    mlflow.log_params(grid_search.best_params_)

    # Manual Log: Metrics
    mlflow.log_metric("accuracy",       accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision",      precision_score(y_test, y_pred))
    mlflow.log_metric("recall",         recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score",       f1_score(y_test, y_pred))
    mlflow.log_metric("roc_auc",        roc_auc_score(y_test, y_prob))
    mlflow.log_metric("best_cv_score",  grid_search.best_score_)

    # Artifact 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    im = ax1.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im)
    ax1.set_title("Confusion Matrix", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14)
    plt.tight_layout()
    mlflow.log_figure(fig1, "confusion_matrix.png")
    plt.close()

    # Artifact 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score   = roc_auc_score(y_test, y_prob)
    fig2, ax2   = plt.subplots(figsize=(6, 5))
    ax2.plot(fpr, tpr, color='steelblue', lw=2, label=f"AUC = {auc_score:.3f}")
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve", fontsize=13, fontweight='bold')
    ax2.legend(loc="lower right")
    plt.tight_layout()
    mlflow.log_figure(fig2, "roc_curve.png")
    plt.close()

    # Classification Report JSON
    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(report_path)
    os.remove(report_path)

    # Log Model
    mlflow.sklearn.log_model(best_model, "model")

    print(f"\nBest Params : {grid_search.best_params_}")
    print(f"Accuracy    : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score    : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC     : {roc_auc_score(y_test, y_prob):.4f}")
    print("Run selesai. Cek MLflow UI di http://127.0.0.1:5000")