# backend/routes/analytics.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"]
)

# ── Summary stats endpoint ───────────────────────────
@router.get("/summary")
async def get_summary():
    """
    Returns overall API usage statistics.
    In Step 9 this will pull from real database.
    """
    return {
        "total_predictions": 0,
        "predictions_today": 0,
        "average_confidence": 0.0,
        "average_response_time_ms": 0.0,
        "most_predicted_class": "N/A",
        "api_uptime_hours": 0.0,
        "status": "Analytics will populate after predictions are made"
    }

# ── Recent predictions endpoint ──────────────────────
@router.get("/recent")
async def get_recent_predictions(limit: int = 10):
    """
    Returns the most recent N predictions made.
    limit: how many records to return (default 10, max 100)
    """
    if limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Limit cannot exceed 100"
        )
    return {
        "predictions": [],
        "count": 0,
        "message": "No predictions yet — make some predictions first"
    }

# ── Model performance endpoint ───────────────────────
@router.get("/model-performance")
async def get_model_performance():
    """
    Returns performance metrics per model version.
    """
    return {
        "models": [
            {
                "version": "v1",
                "total_predictions": 0,
                "average_confidence": 0.0,
                "class_distribution": {
                    "setosa": 0,
                    "versicolor": 0,
                    "virginica": 0
                }
            }
        ]
    }

# ── Daily stats endpoint ─────────────────────────────
@router.get("/daily-stats")
async def get_daily_stats(days: int = 7):
    """
    Returns prediction counts grouped by day.
    days: how many days back to show (default 7)
    """
    if days > 30:
        raise HTTPException(
            status_code=400,
            detail="Maximum 30 days allowed"
        )
    return {
        "period_days": days,
        "daily_data": [],
        "message": "Will populate with real data in Step 9"
    }