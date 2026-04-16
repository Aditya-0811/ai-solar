# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings

# Import routers
from routes.predict   import router as predict_router
from routes.analytics import router as analytics_router
from routes.auth      import router as auth_router

# ── Create app ───────────────────────────────────────
app = FastAPI(
    title=       settings.APP_NAME,
    version=     settings.APP_VERSION,
    description= "AI SaaS API — Production Ready",
    docs_url=    "/docs",
    redoc_url=   "/redoc"
)

# ── CORS ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://runlinc.com",
        "https://www.runlinc.com",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ─────────────────────────────────
app.include_router(predict_router)
app.include_router(analytics_router)
app.include_router(auth_router)

# ── Root ─────────────────────────────────────────────
@app.get("/", tags=["Status"])
def root():
    return {
        "app":      settings.APP_NAME,
        "version":  settings.APP_VERSION,
        "status":   "running",
        "docs":     "/docs"
    }

# ── Health ───────────────────────────────────────────
@app.get("/health", tags=["Status"])
def health_check():
    return {
        "status":  "healthy",
        "version": settings.APP_VERSION
    }

# ── Startup ──────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print(f"\n{'='*50}")
    print(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"{'='*50}")

    # Auto-train model if pkl files are missing
    # This runs on Render first boot automatically
    import os
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "model_v1.pkl"
    )

    if not os.path.exists(model_path):
        print("  Model not found — training now...")
        try:
            import subprocess, sys
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            subprocess.run(
                [sys.executable, os.path.join(root, "ml", "train_model.py")],
                check=True
            )
            print("  Model trained ✅")
        except Exception as e:
            print(f"  Training failed: {e}")
    else:
        print("  Model found ✅")

    try:
        from services.ml_service import preload_all_models
        preload_all_models()
    except Exception as e:
        print(f"  Model load warning: {e}")

    print(f"  ✅ Server ready")
    print(f"{'='*50}\n")