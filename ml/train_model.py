# ml/train_model.py
"""
ML Model Training Script
========================
Trains a Random Forest classifier on the Iris dataset.
Saves the trained model + metadata to /models folder.

Run this from the ai-saas/ root:
    python ml/train_model.py
"""

import os
import pickle
import json
import numpy as np
from datetime import datetime

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ── Configuration ────────────────────────────────────
MODEL_VERSION   = "v1"
MODELS_DIR      = "models"
MODEL_FILENAME  = f"model_{MODEL_VERSION}.pkl"
SCALER_FILENAME = f"scaler_{MODEL_VERSION}.pkl"
META_FILENAME   = f"metadata_{MODEL_VERSION}.json"

def train():
    print("\n" + "="*55)
    print("  AI SaaS — Model Training Pipeline")
    print("="*55)

    # ── Step 1: Load data ────────────────────────────
    print("\n[1/6] Loading dataset...")
    iris = load_iris()
    X = iris.data        # Features: 4 measurements per flower
    y = iris.target      # Labels:   0=setosa, 1=versicolor, 2=virginica

    print(f"      Total samples : {X.shape[0]}")
    print(f"      Features      : {X.shape[1]}")
    print(f"      Classes       : {list(iris.target_names)}")

    # ── Step 2: Split data ───────────────────────────
    print("\n[2/6] Splitting into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y          # Keeps class balance in both splits
    )
    print(f"      Training samples : {X_train.shape[0]}")
    print(f"      Test samples     : {X_test.shape[0]}")

    # ── Step 3: Scale features ───────────────────────
    print("\n[3/6] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("      Scaler fitted ✅")

    # ── Step 4: Train model ──────────────────────────
    print("\n[4/6] Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,       # 100 decision trees
        max_depth=None,         # Trees grow until pure
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1               # Use all CPU cores
    )
    model.fit(X_train_scaled, y_train)
    print("      Model trained ✅")

    # ── Step 5: Evaluate model ───────────────────────
    print("\n[5/6] Evaluating model performance...")

    # Basic accuracy
    y_pred     = model.predict(X_test_scaled)
    accuracy   = accuracy_score(y_test, y_pred)

    # Cross-validation (more reliable than single split)
    cv_scores  = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean    = cv_scores.mean()
    cv_std     = cv_scores.std()

    # Confidence scores for each prediction
    y_proba    = model.predict_proba(X_test_scaled)
    avg_conf   = float(np.max(y_proba, axis=1).mean())

    print(f"\n      Test Accuracy        : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"      CV Score (5-fold)    : {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"      Avg Confidence       : {avg_conf:.4f}")

    print(f"\n      Classification Report:")
    print("      " + "-"*45)
    report = classification_report(
        y_test, y_pred,
        target_names=iris.target_names
    )
    for line in report.split("\n"):
        print(f"      {line}")

    # ── Step 6: Save model ───────────────────────────
    print("\n[6/6] Saving model, scaler and metadata...")

    # Create models/ folder if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save the trained model
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"      Model saved  → {model_path}")

    # Save the scaler (MUST use same scaler at inference time)
    scaler_path = os.path.join(MODELS_DIR, SCALER_FILENAME)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"      Scaler saved → {scaler_path}")

    # Save metadata as JSON
    metadata = {
        "version":          MODEL_VERSION,
        "algorithm":        "RandomForestClassifier",
        "trained_at":       datetime.utcnow().isoformat(),
        "dataset":          "iris",
        "accuracy":         round(accuracy, 4),
        "cv_mean":          round(float(cv_mean), 4),
        "cv_std":           round(float(cv_std), 4),
        "avg_confidence":   round(avg_conf, 4),
        "n_estimators":     100,
        "train_samples":    int(X_train.shape[0]),
        "test_samples":     int(X_test.shape[0]),
        "features": {
            "count": int(X.shape[1]),
            "names": [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width"
            ]
        },
        "classes": {
            "count": 3,
            "names": list(iris.target_names),
            "mapping": {
                "0": "setosa",
                "1": "versicolor",
                "2": "virginica"
            }
        },
        "feature_importances": {
            name: round(float(imp), 4)
            for name, imp in zip(
                ["sepal_length","sepal_width",
                 "petal_length","petal_width"],
                model.feature_importances_
            )
        }
    }

    meta_path = os.path.join(MODELS_DIR, META_FILENAME)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"      Metadata saved → {meta_path}")

    # ── Done ─────────────────────────────────────────
    print("\n" + "="*55)
    print("  ✅ Training complete!")
    print(f"  Accuracy   : {accuracy*100:.2f}%")
    print(f"  CV Score   : {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
    print(f"  Model path : {model_path}")
    print("="*55 + "\n")

    return model, scaler, metadata


if __name__ == "__main__":
    train()