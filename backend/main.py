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
        "http://localhost:8000",
        "http://127.0.0.1:8000",
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

    try:
        from services.ml_service import preload_all_models
        preload_all_models()
    except Exception as e:
        print(f"  ⚠️  Model preload warning: {e}")

    print(f"  ✅ Server ready → http://127.0.0.1:8000")
    print(f"  📖 Docs        → http://127.0.0.1:8000/docs")
    print(f"{'='*50}\n")