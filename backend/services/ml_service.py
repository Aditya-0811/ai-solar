# backend/services/ml_service.py
"""
ML Service
==========
Handles all model loading and prediction logic.
Designed so main.py and routes never touch
pickle/numpy directly — all ML stays here.
"""

import os
import pickle
import json
import numpy as np
from typing import Optional

# ── Path configuration ───────────────────────────────
# Works whether you run from backend/ or ai-saas/
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")

# ── Model registry ───────────────────────────────────
# Stores loaded models in memory — only loads from
# disk once per server startup (much faster)
_model_registry: dict = {}


def _get_model_paths(version: str) -> dict:
    """Returns file paths for a given model version."""
    return {
        "model":    os.path.join(MODELS_DIR, f"model_{version}.pkl"),
        "scaler":   os.path.join(MODELS_DIR, f"scaler_{version}.pkl"),
        "metadata": os.path.join(MODELS_DIR, f"metadata_{version}.json"),
    }


def _load_model(version: str) -> dict:
    """
    Loads model + scaler + metadata from disk.
    Caches in _model_registry so disk is only
    read once per server session.
    """
    # Return from cache if already loaded
    if version in _model_registry:
        return _model_registry[version]

    paths = _get_model_paths(version)

    # Check all files exist before loading
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file missing: {path}\n"
                f"Run: python ml/train_model.py"
            )

    # Load model from pickle
    with open(paths["model"], "rb") as f:
        model = pickle.load(f)

    # Load scaler from pickle
    with open(paths["scaler"], "rb") as f:
        scaler = pickle.load(f)

    # Load metadata from JSON
    with open(paths["metadata"], "r") as f:
        metadata = json.load(f)

    # Cache everything together
    _model_registry[version] = {
        "model":    model,
        "scaler":   scaler,
        "metadata": metadata
    }

    print(f"[ML Service] Model {version} loaded ✅ "
          f"(accuracy: {metadata['accuracy']})")

    return _model_registry[version]


def get_prediction(
    features: list,
    model_version: str = "v1"
) -> dict:
    """
    Core prediction function.
    Called by the /predict route.

    Args:
        features:      List of floats matching model's
                       expected feature count
        model_version: Which model version to use

    Returns:
        dict with prediction, confidence, and metadata
    """
    # Load model (from cache or disk)
    registry_entry = _load_model(model_version)
    model          = registry_entry["model"]
    scaler         = registry_entry["scaler"]
    metadata       = registry_entry["metadata"]

    # ── Validate input ───────────────────────────────
    expected_count = metadata["features"]["count"]
    if len(features) != expected_count:
        raise ValueError(
            f"Expected {expected_count} features, "
            f"got {len(features)}. "
            f"Required: {metadata['features']['names']}"
        )

    # ── Preprocess ───────────────────────────────────
    # Convert list → numpy array with correct shape
    X = np.array(features).reshape(1, -1)

    # Apply same scaling used during training
    X_scaled = scaler.transform(X)

    # ── Predict ──────────────────────────────────────
    # Get predicted class index (0, 1, or 2)
    prediction_idx = model.predict(X_scaled)[0]

    # Get probability for each class
    probabilities  = model.predict_proba(X_scaled)[0]

    # Confidence = probability of the predicted class
    confidence     = float(probabilities[prediction_idx])

    # Convert class index to human-readable label
    class_names    = metadata["classes"]["names"]
    prediction_label = class_names[prediction_idx]

    # ── Build full response ──────────────────────────
    return {
        "prediction":     prediction_label,
        "confidence":     round(confidence, 4),
        "class_index":    int(prediction_idx),
        "all_classes": {
            name: round(float(prob), 4)
            for name, prob in zip(class_names, probabilities)
        },
        "model_version":  model_version,
        "feature_names":  metadata["features"]["names"]
    }


def get_model_info(version: str = "v1") -> dict:
    """
    Returns metadata about a loaded model.
    Used by the /predict/models endpoint.
    """
    registry_entry = _load_model(version)
    return registry_entry["metadata"]


def get_available_versions() -> list:
    """
    Scans the models/ folder and returns
    all versions that have all 3 required files.
    """
    versions = []
    if not os.path.exists(MODELS_DIR):
        return versions

    for fname in os.listdir(MODELS_DIR):
        if fname.startswith("model_") and fname.endswith(".pkl"):
            version = fname.replace("model_", "").replace(".pkl", "")
            paths   = _get_model_paths(version)
            # Only include if all 3 files exist
            if all(os.path.exists(p) for p in paths.values()):
                versions.append(version)

    return sorted(versions)


def preload_all_models():
    """
    Called at server startup to load all models
    into memory before first request arrives.
    Prevents slow first-request latency.
    """
    versions = get_available_versions()
    if not versions:
        print("[ML Service] ⚠️  No models found in models/")
        print("[ML Service]    Run: python ml/train_model.py")
        return

    for version in versions:
        try:
            _load_model(version)
        except Exception as e:
            print(f"[ML Service] ❌ Failed to load {version}: {e}")