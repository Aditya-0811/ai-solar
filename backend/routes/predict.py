# backend/routes/predict.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import time

router = APIRouter(
    prefix="/predict",
    tags=["Predictions"]
)

# ── Request model ────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]
    model_version: Optional[str] = "v1"
    user_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2],
                "model_version": "v1",
                "user_id": "user_123"
            }
        }
    }

# ── Response model ───────────────────────────────────
class PredictResponse(BaseModel):
    prediction:          str
    confidence:          float
    class_index:         int
    all_classes:         Dict[str, float]
    model_version:       str
    processing_time_ms:  float
    input_features:      List[float]
    feature_names:       List[str]

# ── Batch models ─────────────────────────────────────
class BatchPredictRequest(BaseModel):
    items: List[PredictRequest]

class BatchPredictResponse(BaseModel):
    results:        List[PredictResponse]
    total_items:    int
    total_time_ms:  float

# ── Single prediction ────────────────────────────────
@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Send feature values and get AI prediction back.
    """
    start_time = time.time()

    try:
        from services.ml_service import get_prediction

        result = get_prediction(
            features=request.features,
            model_version=request.model_version
        )

        processing_time = (time.time() - start_time) * 1000

        return PredictResponse(
            prediction=         result["prediction"],
            confidence=         result["confidence"],
            class_index=        result["class_index"],
            all_classes=        result["all_classes"],
            model_version=      request.model_version,
            processing_time_ms= round(processing_time, 2),
            input_features=     request.features,
            feature_names=      result["feature_names"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# ── Batch prediction ─────────────────────────────────
@router.post("/batch", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest):
    """
    Send multiple feature sets at once.
    """
    start_time = time.time()
    results    = []

    for item in request.items:
        try:
            from services.ml_service import get_prediction

            result = get_prediction(
                features=item.features,
                model_version=item.model_version
            )
            item_time = (time.time() - start_time) * 1000
            results.append(PredictResponse(
                prediction=         result["prediction"],
                confidence=         result["confidence"],
                class_index=        result["class_index"],
                all_classes=        result["all_classes"],
                model_version=      item.model_version,
                processing_time_ms= round(item_time, 2),
                input_features=     item.features,
                feature_names=      result["feature_names"]
            ))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Batch prediction failed: {str(e)}"
            )

    total_time = (time.time() - start_time) * 1000

    return BatchPredictResponse(
        results=        results,
        total_items=    len(results),
        total_time_ms=  round(total_time, 2)
    )

# ── List models ──────────────────────────────────────
@router.get("/models")
async def list_models():
    """
    Returns all available model versions.
    """
    try:
        from services.ml_service import get_available_versions, get_model_info

        versions  = get_available_versions()
        models    = []

        for v in versions:
            info = get_model_info(v)
            models.append({
                "version":           info.get("version"),
                "algorithm":         info.get("algorithm"),
                "accuracy":          info.get("accuracy"),
                "trained_at":        info.get("trained_at"),
                "features_required": info["features"]["count"],
                "feature_names":     info["features"]["names"],
                "classes":           info["classes"]["names"],
                "status":            "active"
            })

        return {
            "available_models": models,
            "total":            len(models),
            "default_model":    "v1"
        }
    except Exception as e:
        return {
            "available_models": [],
            "total":            0,
            "error":            str(e)
        }